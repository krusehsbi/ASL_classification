import tensorflow as tf

# Load the Keras model
keras_model = tf.keras.models.load_model('models/MobileNet.keras')

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

# Save the converted model to a file
with open('models/MobileNet.tflite', 'wb') as f:
    f.write(tflite_model)
