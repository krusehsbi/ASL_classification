import pandas as pd
import matplotlib.pyplot as plt
import os

# Read data from the CSV file
hist = os.path.join(os.path.dirname(__file__), 'hists/student_teacher_mobilenet_10.csv')


import pandas as pd
import matplotlib.pyplot as plt

# Function to plot data from a single CSV file
def plot_data_from_csv(file_path, color):
    # Read data from the CSV file
    df = pd.read_csv(file_path)
    
    # Generate the 'iteration' column as a sequence starting from 0
    df.insert(0, 'iteration', range(len(df)))

    # Extracting the number
    temp = float(file_path.split('_')[-1].split('.')[0])

    # Plotting
    plt.plot(df['iteration'], df['distillation_loss'], label='Distillation Loss (T={})'.format(temp), color=color, linestyle='-')
    #plt.plot(df['iteration'], df['val_accuracy'], label=f'{file_path} Validation Accuracy', color=color, linestyle='--')
    plt.plot(df['iteration'], df['val_distillation_loss'], label='Validation Distillation Loss (T={})'.format(temp), color=color, linestyle=':')

# List of CSV files to plot
csv_files = [os.path.join(os.path.dirname(__file__), 'hists/student_teacher_mobilenet_5.csv'), 
            os.path.join(os.path.dirname(__file__), 'hists/student_teacher_mobilenet_10.csv')]  # Replace with your actual file names

# Define a list of colors for different models
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# Plot data from each CSV file
plt.figure(figsize=(12, 8))

for i, file_path in enumerate(csv_files):
    color = colors[i % len(colors)]
    plot_data_from_csv(file_path, color)

# Add titles and labels
plt.title('Training and Validation Metrics Over Iterations for Multiple Models')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
