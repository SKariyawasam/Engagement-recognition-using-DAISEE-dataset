#!/usr/import/env python
# coding: utf-8
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.mixed_precision import set_global_policy
from daisee_data_preprocessing import DataPreprocessing
import datetime
import os
import argparse
from tqdm import tqdm

# Enable Mixed Precision for RTX GPUs (faster training, less VRAM)
set_global_policy('mixed_float16')

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def create_log_dir(log_dir, checkpoint_dir):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

def network():
    model = tf.keras.Sequential()
    model.add(kl.InputLayer(input_shape=(224, 224, 3)))
    
    # Pre-trained EfficientNetB0
    efficientnet = EfficientNetB0(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
    efficientnet.trainable = False
    model.add(efficientnet)
    
    model.add(kl.Flatten())
    model.add(kl.Dense(1024, activation='relu'))
    model.add(kl.Dense(256, activation='relu'))
    # Mixed precision requires float32 at the output layer for numeric stability
    model.add(kl.Dense(4, activation='sigmoid', dtype='float32', name='prediction'))
    return model

@tf.function
def macro_f1(y, y_hat, thresh=0.5):
    y = tf.cast(y, tf.float32)
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    return tf.reduce_mean(f1)

@tf.function
def train_step(model, optimizer, loss_fn, x, y, accum_gradients, accumulation_steps):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    
    # Scale loss for gradient accumulation
    scaled_loss = loss_value / tf.cast(accumulation_steps, tf.float32)
    
    # Compute gradients (Keras 3 mixed precision handles loss scaling automatically)
    gradients = tape.gradient(scaled_loss, model.trainable_weights)
    
    # Accumulate gradients
    for i in range(len(accum_gradients)):
        if gradients[i] is not None:
            accum_gradients[i].assign_add(gradients[i])
        
    return loss_value, logits

@tf.function
def apply_acc_gradients(model, optimizer, accum_gradients):
    optimizer.apply_gradients(zip(accum_gradients, model.trainable_weights))
    # Reset accumulators
    for grad in accum_gradients:
        grad.assign(tf.zeros_like(grad))

@tf.function
def test_step(model, x, y):
    logits = model(x, training=False)
    return logits

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['unit_test', 'profile', 'full'], default='full')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--accum_steps', type=int, default=4)
    args = parser.parse_args()

    # Small-To-Large Workflow setup
    if args.mode == 'unit_test':
        print("UNIT TEST MODE: Running on 100MB subset (10 batches)...")
        dataset_limit = 10
        EPOCHS = 1
    elif args.mode == 'profile':
        print("PROFILE MODE: Running on 1GB subset (100 batches)...")
        dataset_limit = 100
        EPOCHS = 2
    else:
        print("FULL MODE: Running on complete dataset...")
        dataset_limit = None
        EPOCHS = 50

    BATCH_SIZE = args.batch_size
    ACCUM_STEPS = args.accum_steps
    LR = 0.005
    checkpoint_dir = 'checkpoints/efficientnet_mixed_prec'
    log_dir = 'logs/efficientnet_mixed_prec'
    create_log_dir(log_dir, checkpoint_dir)

    preprocessing_class = DataPreprocessing()

    def get_dataset(tfrecord_path, is_train=True):
        if not os.path.exists(tfrecord_path):
            print(f"Warning: {tfrecord_path} not found. Creating dummy dataset for testing.")
            # Dummy dataset for when records aren't generated yet
            def gen():
                for _ in range(100):
                    yield (tf.random.normal((224, 224, 3)), tf.random.uniform((4,), maxval=2, dtype=tf.int32))
            ds = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.int32), output_shapes=((224, 224, 3), (4,)))
        else:
            ds = tf.data.TFRecordDataset(tfrecord_path)
            ds = ds.map(preprocessing_class.decode, num_parallel_calls=tf.data.AUTOTUNE)
            if is_train:
                ds = ds.shuffle(1000)
        
        if dataset_limit:
            ds = ds.take(dataset_limit * BATCH_SIZE)
            
        ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return ds

    train_set = get_dataset('tfrecords/train.tfrecords', is_train=True)
    val_set = get_dataset('tfrecords/val.tfrecords', is_train=False)

    model = network()
    
    # Keras 3 Mixed precision handles this natively through the policy
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    
    train_loss_avg = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.Mean()
    val_accuracy = tf.keras.metrics.Mean()

    train_summary_writer = tf.summary.create_file_writer(log_dir)

    # Initialize gradient accumulators
    accum_gradients = [tf.Variable(tf.zeros_like(w), trainable=False) for w in model.trainable_weights]

    # Robust Checkpointing
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(f"Restored from {manager.latest_checkpoint}")
    else:
        print("Initializing from scratch.")

    print(f"Effective Batch Size: {BATCH_SIZE * ACCUM_STEPS} (Batch: {BATCH_SIZE}, Accum Steps: {ACCUM_STEPS})")

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        
        step = 0
        for x_batch_train, y_batch_train in tqdm(train_set):
            loss_value, logits = train_step(model, optimizer, loss_fn, x_batch_train, y_batch_train, accum_gradients, ACCUM_STEPS)
            
            train_loss_avg.update_state(loss_value)
            train_accuracy.update_state(macro_f1(y_batch_train, logits))
            
            step += 1
            if step % ACCUM_STEPS == 0:
                apply_acc_gradients(model, optimizer, accum_gradients)
                ckpt.step.assign_add(1)
            
            # Save checkpoint every 1000 accumulation steps
            if int(ckpt.step) % 1000 == 0 and step % ACCUM_STEPS == 0:
                save_path = manager.save()
                print(f"\\nSaved checkpoint for step {int(ckpt.step)}: {save_path}")

        # Validation
        for x_batch_val, y_batch_val in val_set:
            logits = test_step(model, x_batch_val, y_batch_val)
            val_accuracy.update_state(macro_f1(y_batch_val, logits))

        with train_summary_writer.as_default():
            tf.summary.scalar('Train Loss', train_loss_avg.result(), step=epoch)
            tf.summary.scalar('Train F1 Score', train_accuracy.result(), step=epoch)
            tf.summary.scalar('Val F1 Score', val_accuracy.result(), step=epoch)

        print(f"Train Loss: {train_loss_avg.result():.4f}, Train F1: {train_accuracy.result():.4f}, Val F1: {val_accuracy.result():.4f}")
        
        train_accuracy.reset_state()
        val_accuracy.reset_state()
        train_loss_avg.reset_state()
        
        # Save at end of epoch
        manager.save()
