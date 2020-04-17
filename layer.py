import tensorflow as tf
from tensorflow.keras import layers

def Conv2D(filters, kernel_size, strides=[1, 1], padding='same', activation=None):
    return layers.Conv2D(filters=filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding=padding,
                         activation=activation,
                         kernel_regularizer=)