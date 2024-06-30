import pandas as pd
import os
import tensorflow as tf
from customCNN_v7 import CustomCNN_v7
from student_model import StudentCNN

# Constants
LEARNING_RATE = 0.0001
EPOCHS = 10
BATCH_SIZE = 16
INPUT_SHAPE = (224, 224, 3)
ALPHA = 0.1
TEMPERATURE = 3.0

# Read data
csv_train = os.path.join(os.path.dirname(__file__), 'data/asl_test/data.csv')
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
    img = tf.image.resize(img, INPUT_SHAPE[:2])
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

# Data augmentation
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('vertical'),
    tf.keras.layers.RandomRotation(0.33),
])

# Define and load the teacher model
teacher_model = CustomCNN_v7(img_shape=INPUT_SHAPE)
inputs = teacher_model.input_layer
x = augmentation(inputs)
outputs = teacher_model.call(x)
teacher_model = tf.keras.Model(inputs=inputs, outputs=outputs)
teacher_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
# Assuming that the teacher model has been trained and saved, we load it
teacher_model.load_weights('models/customV7.keras')

# Define the student model
student_model = StudentCNN(img_shape=INPUT_SHAPE)

# Build the student model
student_model.build((None,) + INPUT_SHAPE)

# Print model summary
student_model.summary()

# Custom training step for knowledge distillation
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        student_logits = student_model(images, training=True)
        teacher_logits = teacher_model(images, training=False)
        
        # Student loss (cross-entropy with true labels)
        student_loss = tf.keras.losses.categorical_crossentropy(labels, student_logits, from_logits=False)  # Changed from_logits to False
        
        # Distillation loss (cross-entropy with soft targets)
        soft_targets = tf.nn.softmax(teacher_logits / TEMPERATURE, axis=1)
        distillation_loss = tf.keras.losses.categorical_crossentropy(soft_targets, student_logits / TEMPERATURE, from_logits=False)  # Changed from_logits to False
        
        # Combine the losses
        loss = ALPHA * student_loss + (1 - ALPHA) * distillation_loss
    
    # Compute gradients
    gradients = tape.gradient(loss, student_model.trainable_variables)
    # Apply gradients
    optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))
    
    return loss


# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# Early Stopping Callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Custom training loop
best_val_loss = float('inf')
early_stopping_wait = 0
best_weights = None

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    
    # Training step
    for images, labels in train_dataset:
        loss = train_step(images, labels)
    
    # Validate after each epoch
    val_loss = 0
    val_steps = 0
    for val_images, val_labels in val_dataset:
        student_logits = student_model(val_images, training=False)
        student_loss = tf.keras.losses.categorical_crossentropy(val_labels, student_logits, from_logits=False)  # Changed from_logits to False
        val_loss += tf.reduce_sum(student_loss)
        val_steps += 1
    val_loss /= val_steps
    print(f'Validation loss: {val_loss:.4f}')
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = student_model.get_weights()
        early_stopping_wait = 0
    else:
        early_stopping_wait += 1
        if early_stopping_wait >= early_stopping.patience:
            print('Early stopping triggered')
            break

# Restore best weights
if best_weights is not None:
    student_model.set_weights(best_weights)

# Compile the student model
student_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

# Evaluate the student model
student_model.evaluate(val_dataset)

# Save the student model
student_model.save('models/student_model.keras')