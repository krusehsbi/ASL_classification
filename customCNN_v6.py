import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout, Input, GlobalAveragePooling2D)


class CustomCNN_v6(tf.keras.Model):
    def __init__(self, img_shape):
        super(CustomCNN_v6, self).__init__()
        self.conv1 = Conv2D(32, 3, padding='same', activation='relu')
        self.conv2 = Conv2D(64, 3, padding='same', activation='relu')
        self.pool1 = MaxPooling2D(pool_size=(2, 2))
        self.conv3 = Conv2D(128, 3, padding='same', activation='relu')
        self.conv4 = Conv2D(256, 3, padding='same', activation='relu')
        self.pool2 = MaxPooling2D(pool_size=(2, 2))
        self.conv5 = Conv2D(512, 3, padding='same', activation='relu')
        self.pool3 = MaxPooling2D(pool_size=(2, 2))
        self.avgPooling = GlobalAveragePooling2D()
        self.dense1 = Dense(512, activation='relu')
        self.dropout1 = Dropout(0.5)
        self.dense2 = Dense(256, activation='relu')
        self.dense3 = Dense(27, activation='relu')
        self.input_layer = Input(shape=img_shape)

    def call(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.avgPooling(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
