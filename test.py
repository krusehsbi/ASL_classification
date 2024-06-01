import os
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

BATCH_SIZE = 16
INPUT_SHAPE = (224, 224)
# read test data
csv_test = os.path.join(os.path.dirname(__file__), 'data/synthetic_asl_dataset/Test_Alphabet/data.csv')
df_letter = pd.read_csv(csv_test)

df_letter = df_letter.sample(frac=1)
letter_oh = pd.get_dummies(df_letter['letter']).astype(int)
letter_oh = letter_oh.to_numpy()

# define dataset
ds_test = tf.data.Dataset.from_tensor_slices(
    (df_letter['path'].values, letter_oh)
)


def preprocess(path, letter):
    img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, INPUT_SHAPE)
    return img, letter


# Map preprocessing function to dataset
ds_test = ds_test.map(preprocess)

# batch and prefetch dataset
test_dataset = ds_test.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), "models/customV1.keras"))
model.summary()

eval_test = model.evaluate(test_dataset)

# Losses and Accuracies Test
loss_test = eval_test[0]
accuracy_test = eval_test[1]

df_test = pd.DataFrame({
    'Loss': [loss_test],
    'Accuracy': [accuracy_test]
})

print('Evaluation Test ----------------------------------')
print(df_test)

print('Confusion Matrix ---------------------------------')
# Ensure the test dataset is batched
ds_test = ds_test.batch(32)

# Step 1: Make predictions
predictions = model.predict(ds_test)
predicted_labels = np.argmax(predictions, axis=1)

# Step 2: Extract true labels from the dataset
true_labels = np.concatenate([y for x, y in ds_test], axis=0)

# If your true labels are one-hot encoded, convert them to class indices
if true_labels.ndim > 1:
    true_labels = np.argmax(true_labels, axis=1)

# Step 3: Compute the confusion matrix
conf_matrix = tf.math.confusion_matrix(true_labels, predicted_labels)

# Step 4: Print the confusion matrix
print("Confusion Matrix (TensorFlow):\n", conf_matrix)

# Step 5: Visualize the confusion matrix using Matplotlib and Seaborn
def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    """
    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, cmap=plt.cm.Blues, xticklabels=class_names, yticklabels=class_names, fmt='g')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

# Convert TensorFlow confusion matrix to numpy array
conf_matrix_np = conf_matrix.numpy()

class_names = ['A', 'B', 'Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Plotting the confusion matrix
figure = plot_confusion_matrix(conf_matrix_np, class_names)
plt.show()