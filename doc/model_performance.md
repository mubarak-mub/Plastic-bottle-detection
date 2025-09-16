# Model Performance Documentation

## Overview

This document describes the performance metrics and results of an AI Project – Plastic Bottle Label Detection.
The goal of the project is to classify plastic bottles into two categories:

- Bottle with Label ✅

- Bottle without Label ❌

The trained model is integrated with an ESP32 microcontroller to control LEDs:

- Green LED → Bottle with Label

- Red LED → Bottle without Label

## Model Details

- Algorithm: Convolutional Neural Network (CNN)

- Framework: TensorFlow / Keras

- Input: Images of plastic bottles captured via camera (DroidCam)

## Dataset Split:

- Training: 70%

- Validation: 10%

- Testing: 20%

- Epochs: 8

- Batch Size: 32

- Features Used

The model was trained directly on image data. Each image was preprocessed as follows:

Resizing: All images resized to 224x224 pixels.

Normalization: Pixel values scaled between 0 and 1.

Augmentation: Random flips, rotations, and zoom applied to increase dataset diversity.

## Performance Metrics

Accuracy: 92.5%

Precision (With Label): 100%

Recall (With Label): 100%

F1-Score (With Label): 100%

Precision (Without Label): 100%

Recall (Without Label): 100%

F1-Score (Without Label): 100%

## Key Findings

CNN provided robust performance in classifying bottles with and without labels.

Data augmentation improved the model’s generalization ability.

Integration with ESP32 successfully controlled LEDs in real-time classification.

Green LED lights up for bottles with label; Red LED lights up for bottles without label.

## Recommendations

Collect a larger and more balanced dataset to further improve accuracy.

Test the system in different lighting and background conditions.

Consider deploying the trained model directly on ESP32 with TensorFlow Lite.

Extend the project to detect other types of bottles or packaging.