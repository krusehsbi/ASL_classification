import pandas as pd
import os
import numpy as np
import tensorflow as tf

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
train_size = int(0.9 * len(df_letter))
val_size = int(0.1 * len(df_letter))

train_dataset = ds.take(train_size)
val_dataset = ds.skip(train_size).take(val_size)

# Batch and prefetch datasets
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# Model definition
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE + (3,))

# Freeze first 130 layers
for layer in model.layers[:130]:
    layer.trainable = False

# Add layers on top of MobileNetV2
preprocess_mobilenet = tf.keras.applications.mobilenet_v2.preprocess_input

inputs = tf.keras.Input(shape=INPUT_SHAPE + (3,))

# augment data
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('vertical'),
    tf.keras.layers.RandomRotation(0.33),
])

x = augmentation(inputs)
x = preprocess_mobilenet(x)
x = model(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)

outputs = tf.keras.layers.Dense(27, activation='softmax')(x)

# Define the model
final_model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

# Print model summary
final_model.summary()

# Train the model
history = final_model.fit(train_dataset,
                          epochs=EPOCHS,
                          validation_data=val_dataset)
