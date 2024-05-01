# ASL_classification

Table of contents:

- [Introduction](#introduction)
- [Setup](#setup)
- [Data Inspection](#data-inspection)

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