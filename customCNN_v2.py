import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class CustomCNN_v2(tf.keras.Model):
    def __init__(self, input_shape, num_filters, filter_size, dropout_rate):
        super(CustomCNN_v2, self).__init__()
        self.num_units_fc = 512
        self.conv_layers = []
        for _ in range(7):
            self.conv_layers.append(Conv2D(num_filters, filter_size, activation='relu', padding='same'))
        self.pooling_layer1 = MaxPooling2D(pool_size=(2, 2))
        self.pooling_layer2 = MaxPooling2D(pool_size=(2, 2))
        self.pooling_layer3 = MaxPooling2D(pool_size=(2, 2))
        self.flatten = Flatten()
        self.fc_layer1 = Dense(self.num_units_fc, activation='relu')
        self.dropout1 = Dropout(dropout_rate)
        self.fc_layer2 = Dense(self.num_units_fc, activation='relu')
        self.dropout2 = Dropout(dropout_rate)
        self.fc_layer3 = Dense(self.num_units_fc, activation='relu')
        self.dropout3 = Dropout(dropout_rate)
        self.output_layer = Dense(10, activation='softmax')  # Assuming 10 classes for classification
        
        # Define input shape
        self.input_shape = input_shape

    def call(self, inputs):
        x = inputs
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = self.pooling_layer1(x)
        x = self.pooling_layer2(x)
        x = self.pooling_layer3(x)
        x = self.flatten(x)
        x = self.fc_layer1(x)
        x = self.dropout1(x)
        x = self.fc_layer2(x)
        x = self.dropout2(x)
        x = self.fc_layer3(x)
        x = self.dropout3(x)
        return self.output_layer(x)