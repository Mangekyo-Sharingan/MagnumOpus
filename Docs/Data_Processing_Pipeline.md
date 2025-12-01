# Data Processing Pipeline & Methodology

## 1. Overview
This document details the comprehensive data processing pipeline employed in the Diabetic Retinopathy Classification project. The pipeline is designed to handle large-scale medical imaging datasets (APTOS 2019 and EyePACS), ensuring robust data loading, preprocessing, augmentation, and delivery to deep learning models. The methodology prioritizes reproducibility, memory efficiency, and the handling of class imbalances and varying imaging conditions.

## 2. Data Sources & Ingestion

### 2.1. Datasets
The model training utilizes two primary datasets, merged to create a diverse and extensive training corpus:
*   **APTOS 2019 Blindness Detection Dataset**: Sourced from the Asia Pacific Tele-Ophthalmology Society.
*   **EyePACS (Diabetic Retinopathy Detection) Dataset**: A large-scale dataset provided by EyePACS.

### 2.2. Data Loading Strategy
To handle the substantial volume of high-resolution fundus images without exhausting system memory, a **Lazy Loading** strategy is implemented via the `CustomDataLoader` class.
*   **Metadata Loading**: CSV files containing image filenames and diagnosis labels are loaded into memory as Pandas DataFrames.
*   **On-Demand Image Loading**: Actual image data is read from the disk only when requested during the training or validation steps (via `__getitem__` in the Dataset class).
*   **Library**: OpenCV (`cv2`) is prioritized for image loading due to its performance advantage (2-3x faster) over PIL. The system gracefully falls back to PIL if OpenCV is unavailable or fails for a specific image.

### 2.3. Data Merging & Cleaning
1.  **Standardization**: Column names across datasets are standardized (e.g., mapping `image` to `id_code` and `level` to `diagnosis`) to ensure a unified schema.
2.  **Validation**: The pipeline iterates through the metadata and verifies the existence of every image file on the disk. Entries with missing image files are filtered out to prevent runtime errors.
3.  **Merging**: The cleaned APTOS and EyePACS DataFrames are concatenated into a single `merged_train_df`.
4.  **Shuffling**: The merged dataset is shuffled with a fixed random seed (`random_state=20020315`) to ensure the distribution of data is randomized yet reproducible.

## 3. Data Splitting Methodology

The merged dataset is partitioned into three distinct subsets using a **Stratified Split** strategy to maintain the class distribution (diagnosis severity) across all sets.

*   **Test Set (15%)**: A hold-out set used strictly for final model evaluation. It is isolated immediately to prevent data leakage.
*   **Validation Set (~15%)**: Derived from the remaining data, used for hyperparameter tuning and early stopping during training.
*   **Training Set (~70%)**: The remaining data used for model weight optimization.

**Split Ratios**: 70% Train / 15% Validation / 15% Test.

## 4. Preprocessing Pipelines

Two distinct preprocessing pipelines are implemented to cater to the specific input requirements of the different model architectures.

### 4.1. Pipeline A (VGG16 & ResNet50)
*   **Target Resolution**:
    *   **VGG16**: 224x224 pixels.
    *   **ResNet50**: 256x256 pixels (Enhanced resolution for better feature detection).
*   **Preprocessing Steps**:
    1.  **Resize**: The image is resized such that the longest dimension matches the target size, maintaining the original aspect ratio.
    2.  **Center Padding**: The resized image is pasted onto a square black canvas of the target size, effectively padding the shorter dimension with black borders. This preserves the geometric integrity of the retina without distortion.
    3.  **Normalization**: Pixel values are normalized using standard ImageNet statistics:
        *   Mean: `[0.485, 0.456, 0.406]`
        *   Std: `[0.229, 0.224, 0.225]`

### 4.2. Pipeline B (InceptionV3)
*   **Target Resolution**: 299x299 pixels (or 320x320 as configured).
*   **Preprocessing Steps**: Identical logic to Pipeline A (Resize -> Center Pad -> Normalize) but adapted for the higher resolution requirements of the Inception architecture.

## 5. Data Augmentation (Training Only)

To combat overfitting and improve the model's generalization to varying imaging conditions, an aggressive augmentation policy is applied dynamically during training.

### 5.1. Geometric Augmentations
*   **Random Rotation**: +/- 180 degrees. Fundus images are rotation-invariant; this allows the model to learn features regardless of orientation.
*   **Random Vertical Flip**: Probability $p=0.5$.
*   **Random Horizontal Flip**: Probability $p=0.5$.

### 5.2. Photometric (Color) Augmentations
To simulate differences in camera sensors and lighting conditions:
*   **Color Jitter**:
    *   Brightness: $\pm 20\%$
    *   Contrast: $\pm 20\%$
    *   Saturation: $\pm 20\%$
    *   Hue: $\pm 5\%$

## 6. Input/Output Specifications

### 6.1. Input
*   **Raw Data**: JPEG/PNG images stored in `Data/Aptos/train_images/` and `Data/EyePacs/train/`.
*   **Metadata**: `train.csv` (APTOS) and `trainLabels.csv` (EyePACS).

### 6.2. Output
The pipeline produces PyTorch `DataLoader` objects that yield batches of data:
*   **X (Inputs)**: Tensor of shape `(Batch_Size, 3, Height, Width)`, normalized and augmented (if training).
*   **y (Labels)**: Tensor of shape `(Batch_Size,)` containing integer class labels (0-4).

## 7. Class Definitions
The diagnosis labels correspond to the international clinical diabetic retinopathy scale:
*   **0**: No DR
*   **1**: Mild
*   **2**: Moderate
*   **3**: Severe
*   **4**: Proliferative DR
