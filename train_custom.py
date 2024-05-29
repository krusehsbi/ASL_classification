import pandas as pd
import os
import numpy as np
import tensorflow as tf
import sys
from customCNN_v1 import CustomCNN_v1

LEARNING_RATE = 0.0001
EPOCHS = 20
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
train_size = int(0.9 * len(df_letter))
val_size = int(0.1 * len(df_letter))

train_dataset = ds.take(train_size)
val_dataset = ds.skip(train_size).take(val_size)

# Batch and prefetch datasets
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# Define the model
custom_Model = CustomCNN_v1(input_shape=INPUT_SHAPE + (3,),num_filters=64, filter_size=(3, 3), dropout_rate=0.5)

# Compile the model
custom_Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

# Print model summary
custom_Model.summary()

# Train the model
history = custom_Model.fit(train_dataset,
                          epochs=EPOCHS,
                          validation_data=val_dataset)

model_name = 'MobileNetV2_lr{}_epochs{}_batchSize{}.keras'.format(LEARNING_RATE, EPOCHS, BATCH_SIZE)
model_path = csv_train = os.path.join(os.path.dirname(__file__), 'models/' + model_name)
custom_Model.save(model_path)

hist_name = 'MobileNetV2_lr{}_epochs{}_batchSize{}.csv'.format(LEARNING_RATE, EPOCHS, BATCH_SIZE)
hist_path = csv_train = os.path.join(os.path.dirname(__file__), 'hists/' + hist_name)
hist_df = pd.DataFrame(history.history)
with open(hist_path, mode='w') as file:
    hist_df.to_csv(file)