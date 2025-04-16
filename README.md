# Breast Cancer Segmentation and Classification System

This repository contains a two-step pipeline for processing breast cancer images. The system first segments images to localize regions with cancerous cells, and then classifies the segmented images as benign or malignant using radiomics features and an SVM classifier. The project is built with object‑oriented design and a modular structure, making it easy to maintain and extend.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)

## Overview

This project is divided into two main steps:

1. **Segmentation:**  
   Loads breast cancer images along with their corresponding ground truth masks using a custom PyTorch dataset. It uses a modified DeepLabV3 segmentation model with a custom segmentation head to predict segmentation masks. The training pipeline includes data augmentation, early stopping, K-Fold cross validation, and visualization of the segmentation predictions.

2. **Classification:**  
   After segmentation, radiomic features are extracted from the segmented outputs using PyRadiomics. These features are preprocessed (normalized, outlier-filtered, and reduced via feature selection) and balanced using SMOTE (applied **only** on the training set). An SVM classifier is then trained using GridSearchCV with K-Fold cross validation, and the decision threshold is fine-tuned for optimal performance.

## Project Structure

The repository is organized into separate directories to keep the segmentation and classification pipelines modular and easy to navigate.

```bash
project_folder/
├── segmentation/                
│   ├── dataset.py               # Contains the BreastCancerDataset class for loading images & masks
│   ├── model.py                 # Contains DeepLabHead & SegmentationModelWrapper (model setup)
│   ├── trainer.py               # Contains SegmentationTrainer for training and evaluation
│   ├── cross_validation.py      # Contains CrossValidator for K-Fold cross validation
│   └── main.py                  # Main script to run segmentation experiments
├── classification/              
│   ├── radiomics_extractor.py   # Contains RadiomicsFeatureExtractor class for feature extraction
│   ├── preprocessor.py          # Contains FeaturePreprocessor for normalization, SMOTE, & feature selection
│   ├── svm_classifier.py        # Contains SVMClassifier & ThresholdAdjustedModel for classification
│   └── main.py                  # Main script to run classification experiments
└── README.md                    # This file
```

## Features

- **Segmentation:**
  - Custom PyTorch `Dataset` for loading images and masks.
  - Modified DeepLabV3 with a custom segmentation head for single-channel (grayscale) inputs.
  - Data augmentation using Albumentations.
  - Training with early stopping and K-Fold cross validation.
  - Visualization tools for displaying input images, true masks, and predicted masks.

- **Classification:**
  - Radiomic feature extraction from segmented images using PyRadiomics.
  - A preprocessing pipeline to normalize features, remove outliers, and perform feature selection.
  - SMOTE oversampling applied only on the training set (to avoid synthetic data in the test set).
  - SVM classification using GridSearchCV with K-Fold cross validation.
  - Threshold adjustment for fine-tuning classifier predictions.
  - Evaluation through classification reports, confusion matrices, and performance metrics.
 
## Results

This section summarizes the results obtained for both segmentation and classification pipelines.

### Segmentation Results

**Hyperparameters:**
- Learning Rate: _[0.01]_
- Batch Size: _[16]_
- Epochs: _[65]_
- Optimizer: _[SGD]_

**Performance Metrics:**
- **Validation IoU:** _[80.68%]_
- **Validation Accuracy:** _[95.92%]_

**Graphs:**
- **Train-Test Loss Graph:**  
  ![Loss Graph](path/to/loss_graph.png)  
  _Figure: The training loss and validation loss over the epochs._

**Sample Outputs:**
- **Segmented Images:**  
  - **Input Image:**  
    ![Input Image](path/to/input_image.png)
  - **Ground Truth Mask:**  
    ![Ground Truth Mask](path/to/gt_mask.png)
  - **Predicted Segmentation:**  
    ![Predicted Segmentation](path/to/predicted_mask.png)

### Classification Results

**Hyperparameters:**
- **SVM Parameters:**  
  - C: _[100]_
  - Kernel: _['linear']_

**Performance Metrics:**
- **Sensitivity:** _[96.00%]_
- **Specificity:** _[85.00%]_
- **Accuracy:** _[91.00%]_

**Graphs:**
- **ROC Curve:**  
  ![ROC Curve](path/to/roc_curve.png)  
  _Figure: ROC curve for the best SVM model._
- **Sensitivity-Specificity Trade-off:**  
  ![Trade-off Graph](path/to/tradeoff_graph.png)  
  _Figure: Trade-off between sensitivity and specificity illustrating the decision threshold effects._


## Requirements

- Python 3.7+
- PyTorch & Torchvision
- Albumentations
- SimpleITK
- PyRadiomics
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn
- pandas
- numpy

## Installation

First, clone the repository:

```bash
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
```

Next, create a virtual environment (recommended) and install the required packages:

```bash
pip install -r requirements.txt
```

If needed, install PyRadiomics and SimpleITK separately:

```bash
pip install pyradiomics
pip install SimpleITK
```
