

import tensorflow as tf
import tensorflow.keras.layers as kl

def network():
    model = tf.keras.Sequential()
    model.add(kl.InputLayer(input_shape=(224, 224, 3)))
    
    # Pre-trained EfficientNetB0
    efficientnet = tf.keras.applications.EfficientNetB0(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
    efficientnet.trainable = False
    model.add(efficientnet)
    
    # Flatten
    model.add(kl.Flatten())
    # First FC
    model.add(kl.Dense(1024, activation='relu'))
    # Second Fc
    model.add(kl.Dense(256, activation='relu'))
    # Output FC with sigmoid for multi-label engagement (engaged, bored, frustrated, confused)
    model.add(kl.Dense(4, activation='sigmoid', name='prediction'))
    return model
