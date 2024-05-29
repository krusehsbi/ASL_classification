import pandas as pd
import os
import numpy as np
import tensorflow as tf
import sys
from customCNN_v1 import CustomCNN_v1

LEARNING_RATE = 0.0001
EPOCHS = 10
BATCH_SIZE = 16

INPUT_SHAPE = (224, 224)

# Read data
csv_train = os.path.join(os.path.dirname(__file__), 'data/synthetic_asl_dataset/Train_Alphabet/data.csv')
df_letter = pd.read_csv(csv_train)

# Shuffle data
letters_train = df_letter.sample(frac=1)

# One-hot encode
letter_oh = pd.get_dummies(letters_train['letter']).astype(int)
letter_oh = letter_oh.to_numpy()

# Define dataset
ds = tf.data.Dataset.from_tensor_slices(
    (letters_train['path'].values, letter_oh))


# Define preprocessing function
def preprocess(path, letter):
    img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, INPUT_SHAPE)
    return img, letter


# Map preprocessing function to dataset
ds = ds.map(preprocess)

# Split into train and validation
train_size = int(0.7 * len(df_letter))
val_size = int(0.3 * len(df_letter))

train_dataset = ds.take(train_size)
val_dataset = ds.skip(train_size).take(val_size)

# Batch and prefetch datasets
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)


augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('vertical'),
    tf.keras.layers.RandomRotation(0.33),
])


# Define the model
custom_Model = CustomCNN_v1(img_shape=INPUT_SHAPE + (3,))

# Create the Keras model using the input layer and the output from the call method
inputs = custom_Model.input_layer
x = augmentation(inputs)
outputs = custom_Model.call(x)
custom_Model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
custom_Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

# Print model summary
custom_Model.summary()

#Early Stopping Callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = custom_Model.fit(train_dataset,
                           epochs=EPOCHS,
                           validation_data=val_dataset,
                           callbacks=[early_stopping])

model_name = 'customV1_dropout02.keras'
model_path = csv_train = os.path.join(os.path.dirname(__file__), 'models/' + model_name)
custom_Model.save(model_path)

hist_name = 'customV1_dropout02.csv'
hist_path = os.path.join(os.path.dirname(__file__), 'hists/' + hist_name)
hist_df = pd.DataFrame(history.history)
with open(hist_path, mode='w') as file:
    hist_df.to_csv(file)
