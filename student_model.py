import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Dropout, Input

class StudentCNN(tf.keras.Model):
    def __init__(self, img_shape):
        super(StudentCNN, self).__init__()
        self.conv1 = Conv2D(16, 3, padding='same', activation='relu')
        self.pool1 = AveragePooling2D(pool_size=(2, 2))
        self.conv2 = Conv2D(32, 3, padding='same', activation='relu')
        self.pool2 = AveragePooling2D(pool_size=(2, 2))
        self.flatten = Flatten()
        self.dense1 = Dense(64, activation='relu')
        self.dropout1 = Dropout(0.5)
        self.dense2 = Dense(27, activation='softmax')  # Softmax for classification
        self.input_layer = Input(shape=img_shape)

    def call(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        return x
