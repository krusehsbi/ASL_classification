# ASL_classification

Table of contents:

- [Introduction](#introduction)
- [Setup](#setup)
- [Data Inspection](#data-inspection)
- [Training](#training)
- [Knowledge Distillation](#knowledge-distillation)

## Introduction

Classification of American Sign Language (ASL) images.

The data used in this project originates from the Synthetic ASL Alphabet Dataset. It can be downloaded from https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet.

The repository was created as part of the 'Deep Learning for Computer Vision' course in the summer semester of 2024.

## Setup

To setup the data first download the dataset and extract in a seperate folder called 'data'.\
Run the [dataset_creation.py](dataset_creation.py) in order to rename the data and create the .csv files used for the tensorflow-dataset:

    python3 dataset_creation.py


## Data Inspection

If you wish to inspect the Data, run

    python3 dataset_inspect.py

This will generate a plot with one randomly sampled image for each letter in the alphabet.

## Training

All relevant Training of custom models is defined within the train_custom module. The corresponding 
custom models are defined in their own Class such as CutomCNN_v1,CutomCNN_v2 etc.
Training of the Transfer Learning is being done in the train_MobileNetV2 module.

## Knowledge distillation

The Knowledge distillation was realized as an online-distillation of the best Custom Model (customCNN_v7). The distillation was first realized in the module knowledge_distillation, detailing
the training step and overall distillation process. Later on it was replaced by the more advanced
knowledge_distillation2 module utilizing a seperat Distiller class.
It has proven successfull in Distilling the custom Model in to a much smaller MobileNetV2 model
drastically decreasing the size of the model while maintaining a similar performance.