import matplotlib.pyplot as plt
import pandas as pd
import os

# Read the data from the CSV file
hist = os.path.join(os.path.dirname(__file__), 'hists/customV1.csv')
df = pd.read_csv(hist)
df['epoch'] = df.index

# Plotting
plt.figure(figsize=(14, 8))

# Loss
plt.subplot(2, 2, 1)
plt.plot(df['epoch'], df['loss'], label='Training Loss')
plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

# Accuracy
plt.subplot(2, 2, 2)
plt.plot(df['epoch'], df['accuracy'], label='Training Accuracy')
plt.plot(df['epoch'], df['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()
