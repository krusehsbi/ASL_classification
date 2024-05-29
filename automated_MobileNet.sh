#!/bin/bash

echo "ITERATE DIFFERENT LEARNING RATES"
# List of learning rates to iterate over
learning_rates=(0.01 0.001 0.0001)

# Iterate over each learning rate
for lr in "${learning_rates[@]}"; do
    echo "Running script with learning rate: $lr"
    python3 train_MobileNetV2.py $lr 20 16
done