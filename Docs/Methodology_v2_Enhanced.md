# Methodology V2: Enhanced Approach

## 1. Introduction
This document details the enhanced methodology adopted to improve model performance, specifically targeting the limitations identified in the baseline approach (e.g., class imbalance, limited augmentation, and uniform learning rates). The focus is on maximizing the performance of **ResNet50** and **InceptionV3**.

## 2. Advanced Data Preprocessing & Augmentation

### 2.1. Increased Input Resolution
To better capture minute retinal lesions (such as microaneurysms), the input resolution for ResNet50 was increased.
*   **ResNet50:** Resolution increased from 224x224 to **256x256**.
*   **InceptionV3:** Maintained at **299x299** (native resolution).

### 2.2. Aggressive Data Augmentation
Recognizing that fundus images are rotation-invariant and that lighting conditions vary significantly between cameras, a more aggressive augmentation pipeline was implemented.
*   **Geometric Augmentations:**
    *   **Random Rotation:** Increased from 15 degrees to **180 degrees** (full rotation invariance).
    *   **Random Vertical Flip:** Added (p=0.5), as the retina has no intrinsic "up" or "down".
    *   **Random Horizontal Flip:** Maintained (p=0.5).
*   **Color Augmentations:**
    *   **Color Jitter:** Added to simulate varying exposure and lighting conditions.
        *   Brightness: 0.2
        *   Contrast: 0.2
        *   Saturation: 0.2
        *   Hue: 0.05

## 3. Model Architectures
The core architectures (ResNet50 and InceptionV3) remain consistent with the baseline to ensure comparability, but the training dynamics applied to them have changed significantly.

## 4. Enhanced Training Strategy

### 4.1. Optimizer: AdamW
*   **Optimizer:** Switched from Adam to **AdamW** (Adam with Decoupled Weight Decay).
*   **Rationale:** AdamW provides better generalization performance by decoupling weight decay from the gradient update, which is particularly effective for deep CNNs.
*   **Weight Decay:** Set to **0.01**.

### 4.2. Differential Learning Rates (Fine-Tuning)
During Stage 2 (Fine-Tuning), a differential learning rate strategy is now applied to prevent "catastrophic forgetting" of the pre-trained ImageNet features.
*   **Backbone Learning Rate:** **0.1x** of the base learning rate.
*   **Classifier Head Learning Rate:** **1.0x** of the base learning rate.
*   **Rationale:** The backbone features are already robust; they need only minor adjustments. The classifier head, being specific to this task, requires more aggressive updates.

### 4.3. Weighted Loss Function
To address the severe class imbalance (dominance of "No DR" cases), the loss function was modified to penalize errors on minority classes more heavily.
*   **Criterion:** **Weighted CrossEntropyLoss**.
*   **Class Weights:** Calculated based on the inverse frequency of the classes (approximate distribution):
    *   Class 0 (No DR): **0.5**
    *   Class 1 (Mild): **2.0**
    *   Class 2 (Moderate): **1.0**
    *   Class 3 (Severe): **3.0**
    *   Class 4 (Proliferative): **4.0**
*   **Label Smoothing:** Maintained (if enabled in config) to prevent the model from becoming over-confident.

## 5. Summary of Changes

| Feature | Baseline (V1) | Enhanced (V2) |
| :--- | :--- | :--- |
| **Resolution (ResNet)** | 224x224 | **256x256** |
| **Rotation Augmentation** | +/- 15 degrees | **+/- 180 degrees** |
| **Vertical Flip** | No | **Yes** |
| **Color Jitter** | No | **Yes** |
| **Optimizer** | Adam | **AdamW** |
| **Fine-Tuning LR** | Uniform (Same for all layers) | **Differential (0.1x for Backbone)** |
| **Loss Function** | Standard CrossEntropy | **Weighted CrossEntropy** |
