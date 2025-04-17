# Breast Cancer Segmentation and Classification System

This repository contains a two-step pipeline for processing breast cancer images. The system first segments images to localize regions with possible cancerous cells, and then classifies the segmented images as benign or malignant using radiomics features and an SVM classifier.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)

## Overview

This project is divided into two main steps:

1. **Segmentation:**
   
   1.1. **Loading the Data:**

   - We’ve built a small helper (a “custom Dataset”) so that, when you point it at a folder, it will automatically find each image and its       matching mask (the file with _mask in its name).

   - Every time we ask for a sample, it returns both the raw grayscale image and the binary mask that tells us where the cancerous cells          actually are.
  
   1.2. **The Segmentation Model:**

   - Under the hood we start with DeepLabV3 (a state‑of‑the‑art semantic segmentation network) that’s been pre‑trained on a large dataset.

   - Because our images are single‑channel (grayscale) instead of three‑channel (RGB), we swap out the very first convolution so it only         expects one channel.
     
   - We also replace the final “classifier” part of DeepLab with a small custom block that ends in a single‑channel output—i.e. one number       per pixel predicting “cancer” vs. “no cancer.”
  
   1.3. **Making Training Robust:**

   - *Data Augmentation:* Before each training pass we randomly flip, rotate, warp, or shift the images so the model learns to handle small        variations in cell shape or orientation.

   - *Early Stopping:* We keep an eye on performance on a held‑out validation set. As soon as validation loss stops improving for a set          number of epochs, we halt training—this prevents the model from “overfitting” (memorizing details that don’t generalize).
     
   - *K‑Fold Cross Validation:* Instead of a single train/validation split, we divide the dataset into K chunks. We train K different
     times, each time holding out a different chunk for validation. This gives a more reliable estimate of how well the model will do on
     brand‑new data.


3. **Classification:**
   
   1.1. **Starting with Segmentation Outputs:**

   - Instead of the raw microscope images, we now feed into our next stage the predicted masks (or ground‑truth masks) produced by the segmentation model.

   - Each mask identifies exactly which pixels belong to a cell region. We pair each mask back with its original image to focus our analysis on just those cellular areas.
  
   1.2. **Radiomic Feature Extraction:**

   - Using PyRadiomics, we compute a large set of quantitative features from each image/mask pair—things like texture measures (how speckled or smooth the region is), shape descriptors (compactness, elongation), and intensity statistics (mean brightness, variance).

   - The result is a big table: one row per image, and one column for every radiomic feature plus a “label” column (0 for benign, 1 for malignant).
  
   1.3. **Preprocessing the Feature Table:**

   - *Normalization:* We convert each feature to a z‑score (subtract the mean, divide by the standard deviation) so that features on wildly different scales become comparable.
     
   - *Outlier Filtering:* Some images might have extreme values in many features (perhaps due to artifacts). We drop any case where more than a small fraction of its features are “too far” (e.g. |z|>2.5) from the norm.
  
   - *Train/Test Split:* We split our cleaned table into a training set and a test set before doing anything else—so that nothing we do on the train data ever “peeks” at the test samples.
  
   - *SMOTE Balancing:* Real datasets often have more benign than malignant examples. On the training set only, we apply SMOTE to synthetically up‑sample the minority class (malignant) so that both classes are equally represented during training. The test set remains untouched and purely real.
  
   1.4. **Feature Selection:**

   - Hundreds of radiomic features can overwhelm even a powerful classifier. We use a simple filter like SelectKBest (with a mutual‑information score) to pick the top K features that carry the most signal about benign vs. malignant.

   - This both speeds up training and reduces the risk of overfitting.
  
   1.5. **Training the SVM Classifier:**

   - We initialize an SVC (support‑vector machine) and set up a GridSearchCV over hyperparameters like the penalty term C and kernel type (linear, RBF, etc.).

   - We nest that inside a K‑Fold cross‑validation loop on the training data so that each combination of parameters is evaluated multiple times on different splits.
     
   - We track metrics like accuracy, sensitivity (true‑positive rate), and specificity (true‑negative rate) in each fold, and pick the final model that maximizes sensitivity (to catch as many malignant cases as possible).
  
   1.6. **Final Evaluation:**

   - We run the tuned model on the hold‑out test set—which has never been used for SMOTE, normalization parameters, or feature selection decisions—to get an honest estimate of real‑world performance.

   - We report a full classification report (precision, recall/sensitivity, F1) and show a confusion matrix heatmap so you can see exactly how many benign vs. malignant cases were correctly or incorrectly labeled.
     

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

  <a href="url">
  <img src="https://github.com/ekaraali/Breast_Cancer_Detection_System/blob/main/images/seg_train_loss_curve.png?raw=true" style="width:750px; height:auto;" alt="Loss Curve">
  </a>

**Sample Outputs:**
- **Segmented Images:**  

<table align="center">
  <tr>
    <td align="center">
      <a href="url">
        <img src="https://github.com/ekaraali/Breast_Cancer_Detection_System/blob/main/images/gt.png?raw=true" width=auto alt="Input Image">
      </a><br>
      <strong>Input Image</strong>
    </td>
    <td align="center">
      <a href="url">
        <img src="https://github.com/ekaraali/Breast_Cancer_Detection_System/blob/main/images/true_mask.png?raw=true" width=auto alt="Ground Truth Mask">
      </a><br>
      <strong>Ground Truth Mask</strong>
    </td>
    <td align="center">
      <a href="url">
        <img src="https://github.com/ekaraali/Breast_Cancer_Detection_System/blob/main/images/predicted_mask.png?raw=true" width=auto alt="Predicted Segmentation">
      </a><br>
      <strong>Predicted Segmentation</strong>
    </td>
  </tr>
</table>

<table align="center">
  <tr>
    <td align="center">
      <a href="url">
        <img src="https://github.com/ekaraali/Breast_Cancer_Detection_System/blob/main/images/gt2.png?raw=true" width=285 alt="Input Image">
      </a><br>
      <strong>Input Image</strong>
    </td>
    <td align="center">
      <a href="url">
        <img src="https://github.com/ekaraali/Breast_Cancer_Detection_System/blob/main/images/true_mask2.png?raw=true" width=285 alt="Ground Truth Mask">
      </a><br>
      <strong>Ground Truth Mask</strong>
    </td>
    <td align="center">
      <a href="url">
        <img src="https://github.com/ekaraali/Breast_Cancer_Detection_System/blob/main/images/predicted_mask2.png?raw=true" width=285 alt="Predicted Segmentation">
      </a><br>
      <strong>Predicted Segmentation</strong>
    </td>
  </tr>
</table>



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
    <a href="url">
  <img src="https://github.com/ekaraali/Breast_Cancer_Detection_System/blob/main/images/roc_curve.png?raw=true" style="width:550px; height:auto;" alt="Loss Curve">
  </a>
- **Sensitivity-Specificity Trade-off:**  
    <a href="url">
  <img src="https://github.com/ekaraali/Breast_Cancer_Detection_System/blob/main/images/trade_off.png?raw=true" style="width:550px; height:auto;" alt="Loss Curve">
  </a>


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
