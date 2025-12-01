# Methodology V1: Baseline Approach

## 1. Introduction
This document outlines the initial methodology used for the Diabetic Retinopathy Classification project. The objective was to establish a baseline performance using standard deep learning practices, pre-trained architectures, and basic data augmentation techniques.

## 2. Data Preprocessing & Augmentation

### 2.1. Image Preprocessing
The preprocessing pipeline focused on standardizing the input images from the APTOS and EyePACS datasets.
*   **Resizing:**
    *   **ResNet50 / VGG16:** Images were resized to **224x224** pixels.
    *   **InceptionV3:** Images were resized to **299x299** pixels.
*   **Aspect Ratio Preservation:** Images were resized maintaining their aspect ratio and then centered on a black square canvas to prevent distortion of the retinal features.
*   **Normalization:** Pixel values were normalized using ImageNet statistics:
    *   Mean: `[0.485, 0.456, 0.406]`
    *   Std: `[0.229, 0.224, 0.225]`

### 2.2. Data Augmentation (Training Only)
To prevent overfitting, a standard set of geometric augmentations was applied during training. These were optimized for speed and general applicability.
*   **Random Horizontal Flip:** Applied with a probability of 0.5.
*   **Random Rotation:** Images were rotated by up to **15 degrees**.
*   **Note:** No color augmentation or vertical flipping was applied in this baseline phase.

## 3. Model Architectures

Three Convolutional Neural Network (CNN) architectures were evaluated. All models were pre-trained on the ImageNet dataset (Transfer Learning).

### 3.1. VGG16
*   **Backbone:** VGG16 with batch normalization.
*   **Classifier Head:**
    *   Linear (25088 -> 4096) -> ReLU -> Dropout (0.5)
    *   Linear (4096 -> 1000) -> ReLU -> Dropout (0.4)
    *   Linear (1000 -> 5) (Output Layer)

### 3.2. ResNet50
*   **Backbone:** ResNet50.
*   **Classifier Head:**
    *   Dropout (0.5)
    *   Linear (2048 -> 512)
    *   BatchNorm1d(512) -> ReLU -> Dropout (0.4)
    *   Linear (512 -> 5) (Output Layer)

### 3.3. InceptionV3
*   **Backbone:** InceptionV3 (with auxiliary logits enabled).
*   **Classifier Head:**
    *   Dropout (0.5)
    *   Linear (2048 -> 512)
    *   BatchNorm1d(512) -> ReLU -> Dropout (0.4)
    *   Linear (512 -> 5) (Output Layer)

## 4. Training Strategy

### 4.1. Optimizer & Learning Rate
*   **Optimizer:** **Adam** (Adaptive Moment Estimation).
*   **Learning Rate:** Fixed initial learning rate (typically **1e-3** or **1e-4** depending on the tuning phase).
*   **Scheduler:** `ReduceLROnPlateau` was used to reduce the learning rate by a factor of 0.5 if the validation loss did not improve for 5 epochs.

### 4.2. Loss Function
*   **Criterion:** Standard **CrossEntropyLoss**.
*   **Class Imbalance:** No explicit weighting or sampling strategies were applied to handle the class imbalance (dominance of Class 0) in the loss function itself.

### 4.3. Training Phases
*   **Stage 1 (Head Training):** The backbone weights were frozen, and only the custom classifier head was trained.
*   **Stage 2 (Fine-Tuning):** Specific layers of the backbone were unfrozen (e.g., `layer4` for ResNet, `Mixed_7c` for Inception) and trained alongside the head.
    *   **Note:** The same learning rate was applied to both the backbone and the head during fine-tuning.

## 5. Hardware & Environment
*   **Framework:** PyTorch
*   **Mixed Precision:** Enabled (FP16) via `torch.cuda.amp` (GradScaler) to reduce memory usage and speed up training.
*   **Compute:** NVIDIA GPU (CUDA).
