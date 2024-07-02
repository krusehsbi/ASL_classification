import tensorflow as tf
import os

# Function to load the model and print its details
def print_model_details(model_path):
    # Load the Keras model
    model = tf.keras.models.load_model(model_path)
    
    # Get the number of parameters
    total_params = model.count_params()
    
    # Get the storage size of the model file
    model_size_bytes = os.path.getsize(model_path)
    model_size_mb = model_size_bytes / (1024 * 1024)  # Convert bytes to MB
    
    # Print the model details
    print(f"Model: {model_path}")
    print(f"Total Parameters: {total_params}")
    print(f"Model Size: {model_size_mb:.2f} MB")

# Example usage
model_path = 'models/student_model_mobilenet_10.keras'

print_model_details(model_path)
