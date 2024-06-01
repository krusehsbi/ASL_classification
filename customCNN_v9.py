import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, AveragePooling2D, Flatten, Dense, Dropout, Input, Add, \
    ReLU


class CustomCNN_v9(tf.keras.Model):
    def __init__(self, img_shape):
        super(CustomCNN_v9, self).__init__()
        self.conv1 = Conv2D(32, 3, padding='same', activation=None)
        self.conv2 = Conv2D(32, 3, padding='same', activation=None)  # Adjusted filters to 32
        self.pool1 = AveragePooling2D(pool_size=(2, 2))

        self.conv3 = Conv2D(64, 3, padding='same', activation=None)
        self.conv4 = Conv2D(64, 3, padding='same', activation=None)  # Adjusted filters to 64
        self.pool2 = AveragePooling2D(pool_size=(2, 2))

        self.conv5 = Conv2D(128, 3, padding='same', activation=None)
        self.conv6 = Conv2D(128, 3, padding='same', activation=None)  # Added another conv layer for residual block
        self.pool3 = AveragePooling2D(pool_size=(2, 2))

        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu')
        self.dropout1 = Dropout(0.5)
        self.dense2 = Dense(256, activation='relu')
        self.dense3 = Dense(128, activation='relu')
        self.dense4 = Dense(64, activation='relu')
        self.dense5 = Dense(32, activation='relu')
        self.dense6 = Dense(27, activation='softmax')  # Softmax for classification
        self.input_layer = Input(shape=img_shape)

    def call(self, inputs):
        x = inputs

        # First Conv Block
        x = self.conv1(x)
        x = ReLU()(x)

        residual = x  # Save the residual for the first block
        x = self.conv2(x)
        x = ReLU()(x)

        # Add Residual Connection
        x = Add()([x, residual])
        x = self.pool1(x)

        # Second Conv Block
        x = self.conv3(x)
        x = ReLU()(x)

        residual = x  # Save the residual for the second block
        x = self.conv4(x)
        x = ReLU()(x)

        # Add Residual Connection
        x = Add()([x, residual])
        x = self.pool2(x)

        # Third Conv Block
        x = self.conv5(x)
        x = ReLU()(x)

        residual = x  # Save the residual for the third block
        x = self.conv6(x)
        x = ReLU()(x)

        # Add Residual Connection
        x = Add()([x, residual])
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense6(x)

        return x

