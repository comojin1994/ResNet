from layer import *

import tensorflow as tf
import numpy as np

# Residual Unit
class ResidualUnit(tf.keras.Model):
    def __init__(self, filter_in, filter_out, kernel_size):
        super(ResidualUnit, self).__init__()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = Conv2D(filter_out // 4, (1, 1))
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = Conv2D(filter_out // 4, kernel_size)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv3 = Conv2D(filter_out, (1, 1))

        if filter_in == filter_out:
            self.identity = lambda x: x
        else:
            self.identity = Conv2D(filter_out, (1, 1))

    def call(self, x, training=False, mask=None):
        h = self.bn1(x, training=training)
        h = tf.nn.relu(h)
        h = self.conv1(h)
        h = self.bn2(h, training=training)
        h = tf.nn.relu(h)
        h = self.conv2(h)
        h = self.bn3(h, training=training)
        h = tf.nn.relu(h)
        h = self.conv3(h)
        return self.identity(x) + h

# Residual Layer
class ResidualLayer(tf.keras.Model):
    def __init__(self, filter_in, filters, kernel_size):
        super(ResidualLayer, self).__init__()
        self.sequence = list()
        for f_in, f_out in zip([filter_in] + list(filters), filters):
            self.sequence.append(ResidualUnit(f_in, f_out, kernel_size))

    def call(self, x, training=False, mask=None):
        for unit in self.sequence:
            x = unit(x, training=training)
        return x

# Model
class ResNet(tf.keras.Model):
    def __init__(self):
        super(ResNet, self).__init__(name='ResNet')
        self.conv1 = Conv2D(64, (3, 3), activation='relu') # 32x32x64
        self.pool1 = tf.keras.layers.MaxPool2D((2, 2)) # 16x16x64

        self.res1 = ResidualLayer(64, (256, 256, 256), (3, 3)) # 16x16x256
        self.res2 = ResidualLayer(256, (512, 512, 512, 512), (3, 3)) # 16x16x512
        self.pool2 = tf.keras.layers.MaxPool2D((2, 2)) # 8x8x512

        self.res3 = ResidualLayer(512, (1024, 1024, 1024, 1024, 1024, 1024), (3, 3)) # 8x8x1024
        self.pool3 = tf.keras.layers.MaxPool2D((2, 2)) # 4x4x1024

        self.res4 = ResidualLayer(1024, (2048, 2048, 2048), (3, 3)) # 4x4x2048
        self.glob = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, x, training=False, mask=None):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.res1(x, training=training)
        x = self.res2(x, training=training)
        x = self.pool2(x)
        x = self.res3(x, training=training)
        x = self.pool3(x)
        x = self.res4(x, training=training)
        x = self.glob(x)
        x = tf.keras.layers.Activation('softmax')(x)
        return x

if __name__ == '__main__':
    unit = ResidualUnit(8, 16, (3, 3))
    layer = ResidualLayer(8, (16, 16), (3, 3))
    model = ResNet()
    print(unit, layer, model)
    model.summary()