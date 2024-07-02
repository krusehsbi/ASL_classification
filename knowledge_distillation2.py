import pandas as pd
import os
import tensorflow as tf
from customCNN_v7 import CustomCNN_v7
from student_model import StudentCNN
from student_model2 import StudentCNN2
from distiller import Distiller
from distiller2 import Distiller2
import keras

# Constants
# LEARNING_RATE = 0.0001
EPOCHS = 1000
BATCH_SIZE = 8
INPUT_SHAPE = (224, 224)
ALPHA = 0.1
TEMPERATURE = 10

# Read data
csv_train = os.path.join(os.path.dirname(__file__), 'asl_dataset/data.csv')
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
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomTranslation(0.1, 0.1)
])

# Load Teacher Model  --------------------------------------------------------------------------------------------------
teacher_model = tf.keras.models.load_model('models/customV7.keras')

'''# Define the student model
student_model = StudentCNN2(img_shape=INPUT_SHAPE + (3,))
inputs = student_model.input_layer
x = augmentation(inputs)
outputs = student_model.call(x)
student_model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Initialisiere das Modell mit korrekter Eingabeform
dummy_input = tf.random.uniform((1, *(224, 224, 3)))
_ = student_model(dummy_input)
# Kompiliere das Modell
student_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

# Überprüfe die Modellzusammenfassung
student_model.summary()'''

model = tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=INPUT_SHAPE + (3,))
preprocess_mobilenet = tf.keras.applications.mobilenet_v2.preprocess_input

inputs = tf.keras.Input(shape=INPUT_SHAPE + (3,))

x = augmentation(inputs)
x = preprocess_mobilenet(x)
x = model(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.5)(x)

outputs = tf.keras.layers.Dense(27, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

# Define the model
student_model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Initialize and compile distiller
distiller = Distiller2(student=student_model, teacher=teacher_model)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy'],
    student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=ALPHA,
    temperature=TEMPERATURE,
)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)
history = distiller.fit(train_dataset,
                        epochs=EPOCHS,
                        validation_data=val_dataset,
                        callbacks=[early_stopping])

student = distiller.student
student_model.compile(metrics=["accuracy"])
_, top1_accuracy = student.evaluate(val_dataset)
print(f"Top-1 accuracy on the test set: {round(top1_accuracy * 100, 2)}%")

# Save the student model
distiller.save('models/distiller_model_mobilenet_{}_{}.keras'.format(TEMPERATURE, ALPHA))

distiller.student.save('models/student_model_mobilenet_{}_{}.keras'.format(TEMPERATURE, ALPHA))

hist_name = 'student_teacher_mobilenet_{}_{}.csv'.format(TEMPERATURE, ALPHA)
hist_path = os.path.join(os.path.dirname(__file__), 'hists/' + hist_name)
hist_df = pd.DataFrame(history.history)
with open(hist_path, mode='w') as file:
    hist_df.to_csv(file)
