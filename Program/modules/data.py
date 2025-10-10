"""
Data loading and preprocessing module for diabetic retinopathy classification
"""
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class DataLoader:
    """Class responsible for loading and preprocessing diabetic retinopathy data"""

    def __init__(self, config):
        self.config = config
        self.aptos_train_df = None
        self.aptos_test_df = None
        self.eyepacs_train_df = None
        self.merged_train_df = None  # Combined dataset
        self.pipeline_a = PipelineA()  # VGG16 & ResNet
        self.pipeline_b = PipelineB()  # InceptionV3

    def load_data(self):
        """Load and merge training data from both APTOS and EyePACS datasets"""
        print("Loading APTOS dataset...")
        self._load_aptos_data()

        print("Loading EyePACS dataset...")
        self._load_eyepacs_data()

        print("Merging datasets...")
        self._merge_datasets()

        print(f"Total training samples after merging: {len(self.merged_train_df)}")

    def _load_aptos_data(self):
        """Load APTOS dataset"""
        if self.config.aptos_train_csv.exists():
            self.aptos_train_df = pd.read_csv(self.config.aptos_train_csv)
            # Add image paths and dataset source
            self.aptos_train_df['image_path'] = self.aptos_train_df['id_code'].apply(
                lambda x: str(self.config.aptos_train_images_dir / f"{x}.png")
            )
            self.aptos_train_df['dataset'] = 'APTOS'
            print(f"Loaded {len(self.aptos_train_df)} APTOS training samples")

        if self.config.aptos_test_csv.exists():
            self.aptos_test_df = pd.read_csv(self.config.aptos_test_csv)
            # Add image paths for test data
            self.aptos_test_df['image_path'] = self.aptos_test_df['id_code'].apply(
                lambda x: str(self.config.aptos_test_images_dir / f"{x}.png")
            )
            print(f"Loaded {len(self.aptos_test_df)} APTOS test samples")

    def _load_eyepacs_data(self):
        """Load EyePACS dataset"""
        if self.config.eyepacs_train_csv.exists():
            self.eyepacs_train_df = pd.read_csv(self.config.eyepacs_train_csv)

            # Standardize column names to match APTOS format
            self.eyepacs_train_df = self.eyepacs_train_df.rename(columns={
                'image': 'id_code',
                'level': 'diagnosis'
            })

            # Add image paths and dataset source
            self.eyepacs_train_df['image_path'] = self.eyepacs_train_df['id_code'].apply(
                lambda x: str(self.config.eyepacs_train_images_dir / f"{x}.jpeg")
            )
            self.eyepacs_train_df['dataset'] = 'EyePACS'

            # Filter out images that don't exist
            existing_images = []
            for idx, row in self.eyepacs_train_df.iterrows():
                if Path(row['image_path']).exists():
                    existing_images.append(idx)

            self.eyepacs_train_df = self.eyepacs_train_df.loc[existing_images]
            print(f"Loaded {len(self.eyepacs_train_df)} EyePACS training samples")

    def _merge_datasets(self):
        """Merge APTOS and EyePACS datasets"""
        datasets_to_merge = []

        if self.aptos_train_df is not None:
            datasets_to_merge.append(self.aptos_train_df)

        if self.eyepacs_train_df is not None:
            datasets_to_merge.append(self.eyepacs_train_df)

        if datasets_to_merge:
            # Combine both datasets
            self.merged_train_df = pd.concat(datasets_to_merge, ignore_index=True)

            # Shuffle the merged dataset
            self.merged_train_df = self.merged_train_df.sample(
                frac=1,
                random_state=self.config.random_state
            ).reset_index(drop=True)

            # Print dataset statistics
            self._print_dataset_statistics()
        else:
            raise ValueError("No training data found!")

    def _print_dataset_statistics(self):
        """Print statistics about the merged dataset"""
        print("\n=== Dataset Statistics ===")

        # Overall statistics
        total_samples = len(self.merged_train_df)
        print(f"Total training samples: {total_samples}")

        # Dataset distribution
        dataset_counts = self.merged_train_df['dataset'].value_counts()
        print("\nDataset distribution:")
        for dataset, count in dataset_counts.items():
            percentage = (count / total_samples) * 100
            print(f"  {dataset}: {count} samples ({percentage:.1f}%)")

        # Diagnosis distribution
        diagnosis_counts = self.merged_train_df['diagnosis'].value_counts().sort_index()
        print("\nDiagnosis distribution:")
        for diagnosis, count in diagnosis_counts.items():
            percentage = (count / total_samples) * 100
            print(f"  Level {diagnosis}: {count} samples ({percentage:.1f}%)")

        # Diagnosis distribution by dataset
        print("\nDiagnosis by dataset:")
        for dataset in self.merged_train_df['dataset'].unique():
            subset = self.merged_train_df[self.merged_train_df['dataset'] == dataset]
            print(f"\n  {dataset}:")
            diag_counts = subset['diagnosis'].value_counts().sort_index()
            for diagnosis, count in diag_counts.items():
                percentage = (count / len(subset)) * 100
                print(f"    Level {diagnosis}: {count} samples ({percentage:.1f}%)")

    def get_train_data(self):
        """Get the merged training dataset"""
        return self.merged_train_df

    def get_test_data(self):
        """Get the APTOS test dataset (EyePACS doesn't have labeled test data)"""
        return self.aptos_test_df

    def get_preprocessor(self, model_type):
        """Get appropriate preprocessor based on model type"""
        if model_type.lower() in ['vgg16', 'resnet50']:
            return self.pipeline_a
        elif model_type.lower() == 'inceptionv3':
            return self.pipeline_b
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def create_data_generators(self, model_type, is_training=True):
        """Create data generators for training and validation"""
        preprocessor = self.get_preprocessor(model_type)
        return preprocessor.create_generators(
            self.train_df,
            self.config,
            is_training=is_training
        )

class PipelineA:
    """Pipeline A: Preprocessing for VGG16 & ResNet (224x224)"""

    def __init__(self):
        self.target_size = 224

    def preprocess_image(self, image_path, is_training=False):
        """
        Process image according to Pipeline A specifications

        Steps:
        1. Load original image
        2. Resize with aspect ratio preservation
        3. Create square canvas (224x224)
        4. Paste and center image
        5. Normalize pixel values
        6. Apply augmentation (training only)
        """
        # Step 1: Load the original image
        image = Image.open(image_path)

        # Step 2: Resize with aspect ratio preservation
        image = self._resize_with_aspect_ratio(image, self.target_size)

        # Step 3 & 4: Create square canvas and paste centered image
        image = self._create_square_canvas_and_center(image, self.target_size)

        # Step 5: Normalize pixel values
        image_array = np.array(image, dtype=np.float32) / 255.0

        # Step 6: Apply augmentation (training only)
        if is_training:
            image_array = self._apply_augmentation(image_array)

        return image_array

    def _resize_with_aspect_ratio(self, image, target_size):
        """Resize image maintaining aspect ratio"""
        # Get original dimensions
        width, height = image.size

        # Find the longest side
        longest_side = max(width, height)

        # Calculate scaling factor
        scale_factor = target_size / longest_side

        # Calculate new dimensions
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Resize image
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _create_square_canvas_and_center(self, image, canvas_size):
        """Create square canvas and center the image"""
        # Create black canvas
        canvas = Image.new('RGB', (canvas_size, canvas_size), (0, 0, 0))

        # Calculate position to center the image
        img_width, img_height = image.size
        x_offset = (canvas_size - img_width) // 2
        y_offset = (canvas_size - img_height) // 2

        # Paste image onto canvas
        canvas.paste(image, (x_offset, y_offset))

        return canvas

    def _apply_augmentation(self, image_array):
        """Apply data augmentation for training"""
        # This is a placeholder - actual augmentation will be handled by ImageDataGenerator
        return image_array

    def create_generators(self, train_df, config, is_training=True):
        """Create data generators with Pipeline A preprocessing"""
        if is_training:
            # Training generator with augmentation
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                fill_mode='constant',
                cval=0,  # Fill with black
                validation_split=config.validation_split,
                preprocessing_function=lambda x: self._preprocess_for_generator(x, self.target_size)
            )

            # Validation generator without augmentation
            val_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=config.validation_split,
                preprocessing_function=lambda x: self._preprocess_for_generator(x, self.target_size)
            )

            return train_datagen, val_datagen
        else:
            # Test generator without augmentation
            test_datagen = ImageDataGenerator(
                rescale=1./255,
                preprocessing_function=lambda x: self._preprocess_for_generator(x, self.target_size)
            )
            return test_datagen

    def _preprocess_for_generator(self, image, target_size):
        """Preprocessing function for ImageDataGenerator"""
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))

        # Apply Pipeline A preprocessing
        image = self._resize_with_aspect_ratio(image, target_size)
        image = self._create_square_canvas_and_center(image, target_size)

        return np.array(image)

class PipelineB:
    """Pipeline B: Preprocessing for InceptionV3 (299x299)"""

    def __init__(self):
        self.target_size = 299

    def preprocess_image(self, image_path, is_training=False):
        """
        Process image according to Pipeline B specifications

        Steps:
        1. Load original image
        2. Resize with aspect ratio preservation
        3. Create square canvas (299x299)
        4. Paste and center image
        5. Normalize pixel values
        6. Apply augmentation (training only)
        """
        # Step 1: Load the original image
        image = Image.open(image_path)

        # Step 2: Resize with aspect ratio preservation
        image = self._resize_with_aspect_ratio(image, self.target_size)

        # Step 3 & 4: Create square canvas and paste centered image
        image = self._create_square_canvas_and_center(image, self.target_size)

        # Step 5: Normalize pixel values
        image_array = np.array(image, dtype=np.float32) / 255.0

        # Step 6: Apply augmentation (training only)
        if is_training:
            image_array = self._apply_augmentation(image_array)

        return image_array

    def _resize_with_aspect_ratio(self, image, target_size):
        """Resize image maintaining aspect ratio"""
        # Get original dimensions
        width, height = image.size

        # Find the longest side
        longest_side = max(width, height)

        # Calculate scaling factor
        scale_factor = target_size / longest_side

        # Calculate new dimensions
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Resize image
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _create_square_canvas_and_center(self, image, canvas_size):
        """Create square canvas and center the image"""
        # Create black canvas
        canvas = Image.new('RGB', (canvas_size, canvas_size), (0, 0, 0))

        # Calculate position to center the image
        img_width, img_height = image.size
        x_offset = (canvas_size - img_width) // 2
        y_offset = (canvas_size - img_height) // 2

        # Paste image onto canvas
        canvas.paste(image, (x_offset, y_offset))

        return canvas

    def _apply_augmentation(self, image_array):
        """Apply data augmentation for training"""
        # This is a placeholder - actual augmentation will be handled by ImageDataGenerator
        return image_array

    def create_generators(self, train_df, config, is_training=True):
        """Create data generators with Pipeline B preprocessing"""
        if is_training:
            # Training generator with augmentation
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                fill_mode='constant',
                cval=0,  # Fill with black
                validation_split=config.validation_split,
                preprocessing_function=lambda x: self._preprocess_for_generator(x, self.target_size)
            )

            # Validation generator without augmentation
            val_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=config.validation_split,
                preprocessing_function=lambda x: self._preprocess_for_generator(x, self.target_size)
            )

            return train_datagen, val_datagen
        else:
            # Test generator without augmentation
            test_datagen = ImageDataGenerator(
                rescale=1./255,
                preprocessing_function=lambda x: self._preprocess_for_generator(x, self.target_size)
            )
            return test_datagen

    def _preprocess_for_generator(self, image, target_size):
        """Preprocessing function for ImageDataGenerator"""
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))

        # Apply Pipeline B preprocessing
        image = self._resize_with_aspect_ratio(image, target_size)
        image = self._create_square_canvas_and_center(image, target_size)

        return np.array(image)

class DataPreprocessor:
    """Class for additional image preprocessing operations"""

    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size

    def resize_image(self, image):
        """Resize image to target size"""
        return cv2.resize(image, self.image_size)

    def normalize_image(self, image):
        """Normalize image pixel values"""
        return image.astype(np.float32) / 255.0

    def enhance_contrast(self, image):
        """Enhance image contrast for better feature extraction"""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(image.shape) == 3:
            # Convert to LAB color space and apply CLAHE to L channel
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            return clahe.apply(image)
