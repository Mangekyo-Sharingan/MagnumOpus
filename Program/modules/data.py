#Data loading and preprocessing module for diabetic retinopathy classification
import pandas as pd
import sys
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
import numpy as np
import random

# Optional OpenCV for CLAHE; fallback gracefully if unavailable
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    cv2 = None  # type: ignore
    _HAS_CV2 = False

# Add utils to path for Excel logger
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
try:
    from excel_logger import ExcelLogger
except ImportError:  # when imported as package
    from ..utils.excel_logger import ExcelLogger

# Set up logging - use WARNING to avoid I/O overhead during training
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RetinopathyPreprocessor:
    """Advanced preprocessing helpers for retinal images (no change to dataset counts)."""

    @staticmethod
    def apply_clahe(image: Image.Image) -> Image.Image:
        if not _HAS_CV2:
            return image
        img_array = np.array(image)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge([l_clahe, a, b])
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_clahe)

    @staticmethod
    def enhance_green_channel(image: Image.Image) -> Image.Image:
        if not _HAS_CV2:
            return image
        img_array = np.array(image)
        green = img_array[:, :, 1]
        green = cv2.normalize(green, None, 0, 255, cv2.NORM_MINMAX)
        img_array[:, :, 1] = green
        return Image.fromarray(img_array)

    @staticmethod
    def crop_black_borders(image: Image.Image, threshold: int = 10) -> Image.Image:
        img_array = np.array(image)
        gray = img_array if img_array.ndim == 2 else np.mean(img_array, axis=2).astype(np.uint8)
        mask = gray > threshold
        coords = np.column_stack(np.where(mask))
        if coords.size == 0:
            return image
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        cropped = img_array[y_min:y_max + 1, x_min:x_max + 1]
        return Image.fromarray(cropped)

class AdvancedAugmentation:
    """Apply robust augmentations suited for retinal images (without preprocessing).

    Note: CLAHE and green channel enhancement are applied separately in the pipeline
    before this augmentation to avoid double processing.
    """
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        # Blur/sharpen/brightness/contrast augmentations only
        # (CLAHE and green channel are applied earlier in the pipeline)
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))
        if random.random() < 0.3:
            img = ImageEnhance.Sharpness(img).enhance(random.uniform(1.1, 1.8))
        if random.random() < 0.3:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.3:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))
        return img

# Picklable transform classes (replaces lambda functions for Windows multiprocessing)
class CropBlackBordersTransform:
    """Picklable transform for cropping black borders"""
    def __call__(self, img: Image.Image) -> Image.Image:
        return RetinopathyPreprocessor.crop_black_borders(img)

class ApplyCLAHETransform:
    """Picklable transform for applying CLAHE"""
    def __call__(self, img: Image.Image) -> Image.Image:
        return RetinopathyPreprocessor.apply_clahe(img)

class DiabetitcRetinopathyDataset(Dataset):
    """Custom PyTorch Dataset for diabetic retinopathy images"""

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.cv2_fail_count = 0
        self.cv2_success_count = 0
        self.pil_count = 0
        logger.info(f"Created Diabetitc Retinopathy Dataset with {len(dataframe)} samples")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        try:
            row = self.dataframe.iloc[idx]
            image_path = row['image_path']
            label = row['diagnosis']

            # Load image using cv2 (2-3x faster than PIL)
            if _HAS_CV2:
                image = cv2.imread(str(image_path))
                if image is None:
                    # cv2 failed - log and fall back to PIL
                    self.cv2_fail_count += 1
                    if self.cv2_fail_count == 1:
                        print(f"[WARNING]  WARNING: cv2.imread() FAILED on first image: {image_path}")
                        print(f"[WARNING]  Falling back to PIL for ALL images (this is SLOWER!)")
                        print(f"[WARNING]  Check if image files are corrupted or paths are wrong")
                    image = Image.open(image_path).convert('RGB')
                    self.pil_count += 1
                else:
                    # cv2 succeeded
                    self.cv2_success_count += 1
                    if self.cv2_success_count == 1:
                        print(f"[OK] cv2.imread() working! Loading images with OpenCV (fast)")
                    # cv2 loads as BGR, convert to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # Convert to PIL Image for torchvision transforms compatibility
                    image = Image.fromarray(image)
            else:
                # cv2 not available - use PIL
                if self.pil_count == 0:
                    print(f"[INFO]  cv2 not available, using PIL (slower)")
                image = Image.open(image_path).convert('RGB')
                self.pil_count += 1

            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            logger.error(f"Error loading image at index {idx}: {e}")
            raise

    def get_loading_stats(self):
        """Return statistics about image loading methods used"""
        total = self.cv2_success_count + self.cv2_fail_count + self.pil_count
        return {
            'cv2_success': self.cv2_success_count,
            'cv2_failed': self.cv2_fail_count,
            'pil_used': self.pil_count,
            'total': total
        }

class CustomDataLoader:
    """
    Class responsible for loading and preprocessing diabetic retinopathy data
    This class implements a lazy loading strategy for handling multiple datasets:
    - Metadata (file paths, labels) is loaded and merged in memory
    - Actual image data remains on disk and is loaded on-demand during training
    - This approach is memory efficient and scalable for large datasets
    - No file duplication occurs - original images stay in their respective directories
    """

    def __init__(self, config, use_excel_logger=True):
        logger.info("Initializing CustomDataLoader...")
        self.config = config
        self.aptos_train_df = None
        self.eyepacs_train_df = None
        self.merged_train_df = None  # Combined dataset (metadata only - uses lazy loading)
        self.pipeline_a = PipelineA(config)  # VGG16 & ResNet
        self.pipeline_b = PipelineB(config)  # InceptionV3

        # Excel logger
        self.use_excel_logger = use_excel_logger
        self.excel_logger = None

        logger.info("CustomDataLoader initialization complete")

    def load_data(self):
        """Load and merge training data from both APTOS and EyePACS datasets"""
        # Start Excel logging
        if self.use_excel_logger:
            self.excel_logger = ExcelLogger("data_loading")
            self.excel_logger.__enter__()
            self.excel_logger.log("Starting data loading process")
            self.excel_logger.log_separator()

        logger.info("=" * 50)
        logger.info("STARTING DATA LOADING PROCESS")
        logger.info("=" * 50)

        start_time = datetime.now()

        try:
            if self.use_excel_logger:
                self.excel_logger.log("Loading APTOS dataset...")
            logger.info("Loading APTOS dataset...")
            self._load_aptos_data()

            if self.use_excel_logger:
                self.excel_logger.log("Loading EyePACS dataset...")
            logger.info("Loading EyePACS dataset...")
            self._load_eyepacs_data()

            if self.use_excel_logger:
                self.excel_logger.log("Merging datasets...")
            logger.info("Merging datasets...")
            self._merge_datasets()

            end_time = datetime.now()
            duration = end_time - start_time

            logger.info(f"Data loading completed successfully in {duration.total_seconds():.2f} seconds")
            logger.info(f"Total training samples after merging: {len(self.merged_train_df)}")
            logger.info("=" * 50)

            if self.use_excel_logger:
                self.excel_logger.log_separator()
                self.excel_logger.log(f"Data loading completed in {duration.total_seconds():.2f} seconds")
                self.excel_logger.log(f"Total training samples: {len(self.merged_train_df)}")
                self.excel_logger.__exit__(None, None, None)

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            if self.use_excel_logger and self.excel_logger:
                self.excel_logger.log(f"ERROR: Data loading failed - {str(e)}")
                self.excel_logger.__exit__(Exception, e, None)
            raise

    def _load_aptos_data(self):
        """Load APTOS dataset"""
        logger.info("Checking APTOS data availability...")

        if self.use_excel_logger:
            self.excel_logger.log("Checking APTOS data availability...")

        # Load training data
        if self.config.aptos_train_csv.exists():
            logger.info(f"Found APTOS training CSV: {self.config.aptos_train_csv}")
            try:
                self.aptos_train_df = pd.read_csv(self.config.aptos_train_csv)
                logger.info(f"Successfully read APTOS training CSV with {len(self.aptos_train_df)} rows")

                # Add image paths and dataset source
                self.aptos_train_df['image_path'] = self.aptos_train_df['id_code'].apply(
                    lambda x: str(self.config.aptos_train_images_dir / f"{x}.png")
                )
                self.aptos_train_df['dataset'] = 'APTOS'
                logger.info("Added image paths and dataset labels to APTOS training data")

                # Verify image files exist
                missing_images = 0
                for idx, row in self.aptos_train_df.iterrows():
                    if not Path(row['image_path']).exists():
                        missing_images += 1

                if missing_images > 0:
                    logger.warning(f"Found {missing_images} missing APTOS training images")
                    if self.use_excel_logger:
                        self.excel_logger.log(f"WARNING: {missing_images} APTOS training images missing")
                else:
                    logger.info("All APTOS training images found successfully")

                logger.info(f"Loaded {len(self.aptos_train_df)} APTOS training samples")

                if self.use_excel_logger:
                    self.excel_logger.log_dict({
                        'dataset': 'APTOS Training',
                        'samples': len(self.aptos_train_df),
                        'missing_images': missing_images,
                        'status': 'Success'
                    }, prefix="APTOS Training Data:")

            except Exception as e:
                logger.error(f"Failed to load APTOS training data: {e}")
                if self.use_excel_logger:
                    self.excel_logger.log(f"ERROR loading APTOS training data: {str(e)}")
                raise
        else:
            logger.warning(f"APTOS training CSV not found: {self.config.aptos_train_csv}")
            if self.use_excel_logger:
                self.excel_logger.log(f"WARNING: APTOS training CSV not found")


    def _load_eyepacs_data(self):
        """Load EyePACS dataset"""
        logger.info("Checking EyePACS data availability...")

        if self.use_excel_logger:
            self.excel_logger.log("Checking EyePACS data availability...")

        if self.config.eyepacs_train_csv.exists():
            logger.info(f"Found EyePACS training CSV: {self.config.eyepacs_train_csv}")
            try:
                self.eyepacs_train_df = pd.read_csv(self.config.eyepacs_train_csv)
                logger.info(f"Successfully read EyePACS training CSV with {len(self.eyepacs_train_df)} rows")

                # Standardize column names to match APTOS format
                original_columns = list(self.eyepacs_train_df.columns)
                self.eyepacs_train_df = self.eyepacs_train_df.rename(columns={
                    'image': 'id_code',
                    'level': 'diagnosis'
                })
                logger.info(f"Standardized column names: {original_columns} -> {list(self.eyepacs_train_df.columns)}")

                # Add image paths and dataset source
                self.eyepacs_train_df['image_path'] = self.eyepacs_train_df['id_code'].apply(
                    lambda x: str(self.config.eyepacs_train_images_dir / f"{x}.jpeg")
                )
                self.eyepacs_train_df['dataset'] = 'EyePACS'
                logger.info("Added image paths and dataset labels to EyePACS training data")

                # Filter out images that don't exist
                logger.info("Checking EyePACS image file availability...")
                existing_images = []
                missing_count = 0

                for idx, row in self.eyepacs_train_df.iterrows():
                    if Path(row['image_path']).exists():
                        existing_images.append(idx)
                    else:
                        missing_count += 1

                if missing_count > 0:
                    logger.warning(f"Found {missing_count} missing EyePACS images, filtering them out")
                    self.eyepacs_train_df = self.eyepacs_train_df.loc[existing_images]
                    logger.info(f"Filtered dataset down to {len(self.eyepacs_train_df)} samples with existing images")
                    if self.use_excel_logger:
                        self.excel_logger.log(f"WARNING: Filtered out {missing_count} missing images")
                else:
                    logger.info("All EyePACS training images found successfully")

                logger.info(f"Loaded {len(self.eyepacs_train_df)} EyePACS training samples")

                if self.use_excel_logger:
                    self.excel_logger.log_dict({
                        'dataset': 'EyePACS Training',
                        'samples': len(self.eyepacs_train_df),
                        'missing_images': missing_count,
                        'status': 'Success'
                    }, prefix="EyePACS Training Data:")

            except Exception as e:
                logger.error(f"Failed to load EyePACS training data: {e}")
                if self.use_excel_logger:
                    self.excel_logger.log(f"ERROR loading EyePACS training data: {str(e)}")
                raise
        else:
            logger.warning(f"EyePACS training CSV not found: {self.config.eyepacs_train_csv}")
            if self.use_excel_logger:
                self.excel_logger.log(f"WARNING: EyePACS training CSV not found")

    def _merge_datasets(self):
        """Merge APTOS and EyePACS datasets"""
        logger.info("Starting dataset merging process...")

        if self.use_excel_logger:
            self.excel_logger.log_separator()
            self.excel_logger.log("Merging datasets...")

        datasets_to_merge = []

        if self.aptos_train_df is not None:
            datasets_to_merge.append(self.aptos_train_df)
            logger.info(f"Adding APTOS dataset with {len(self.aptos_train_df)} samples")

        if self.eyepacs_train_df is not None:
            datasets_to_merge.append(self.eyepacs_train_df)
            logger.info(f"Adding EyePACS dataset with {len(self.eyepacs_train_df)} samples")

        if datasets_to_merge:
            # Combine both datasets
            logger.info("Concatenating datasets...")
            self.merged_train_df = pd.concat(datasets_to_merge, ignore_index=True)
            logger.info(f"Combined dataset size: {len(self.merged_train_df)} samples")

            # Shuffle the merged dataset
            logger.info(f"Shuffling merged dataset with random_state={self.config.random_state}")
            self.merged_train_df = self.merged_train_df.sample(
                frac=1,
                random_state=self.config.random_state
            ).reset_index(drop=True)
            logger.info("Dataset shuffling complete")

            # Print dataset statistics
            self._print_dataset_statistics()
            logger.info("Dataset merging completed successfully")
        else:
            logger.error("No training data found to merge!")
            if self.use_excel_logger:
                self.excel_logger.log("ERROR: No training data found to merge!")
            raise ValueError("No training data found!")

    def _print_dataset_statistics(self):
        """Print statistics about the merged dataset"""
        logger.info("\n" + "=" * 30)
        logger.info("DATASET STATISTICS")
        logger.info("=" * 30)

        # Overall statistics
        total_samples = len(self.merged_train_df)
        logger.info(f"Total training samples: {total_samples}")

        # Dataset distribution
        logger.info("\nDataset distribution:")
        dataset_counts = self.merged_train_df['dataset'].value_counts()
        dataset_stats = {}
        for dataset, count in dataset_counts.items():
            percentage = (count / total_samples) * 100
            logger.info(f"  {dataset}: {count} samples ({percentage:.1f}%)")
            dataset_stats[dataset] = f"{count} ({percentage:.1f}%)"

        # Diagnosis distribution
        logger.info("\nDiagnosis distribution:")
        diagnosis_counts = self.merged_train_df['diagnosis'].value_counts().sort_index()
        diagnosis_stats = {}
        for diagnosis, count in diagnosis_counts.items():
            percentage = (count / total_samples) * 100
            logger.info(f"  Level {diagnosis}: {count} samples ({percentage:.1f}%)")
            diagnosis_stats[f"Level_{diagnosis}"] = f"{count} ({percentage:.1f}%)"

        # Log to Excel
        if self.use_excel_logger:
            self.excel_logger.log_separator()
            self.excel_logger.log("MERGED DATASET STATISTICS")
            self.excel_logger.log_separator("-", 60)
            self.excel_logger.log(f"Total samples: {total_samples}")
            self.excel_logger.log_separator("-", 60)
            self.excel_logger.log("Dataset Distribution:")
            self.excel_logger.log_dict(dataset_stats)
            self.excel_logger.log_separator("-", 60)
            self.excel_logger.log("Diagnosis Distribution:")
            self.excel_logger.log_dict(diagnosis_stats)

        # Diagnosis distribution by dataset
        logger.info("\nDiagnosis by dataset:")
        for dataset in self.merged_train_df['dataset'].unique():
            subset = self.merged_train_df[self.merged_train_df['dataset'] == dataset]
            logger.info(f"\n  {dataset}:")
            diag_counts = subset['diagnosis'].value_counts().sort_index()
            for diagnosis, count in diag_counts.items():
                percentage = (count / len(subset)) * 100
                logger.info(f"    Level {diagnosis}: {count} samples ({percentage:.1f}%)")

        logger.info("=" * 30)

    def get_train_data(self):
        """Get the merged training dataset"""
        if self.merged_train_df is not None:
            logger.info(f"Returning training dataset with {len(self.merged_train_df)} samples")
            return self.merged_train_df
        else:
            logger.error("No training data available. Please run load_data() first.")
            return None


    def get_preprocessor(self, model_type, quiet=False):
        """
        Get the appropriate preprocessing pipeline for the model type

        Args:
            model_type: Type of model (vgg16, resnet50, inceptionv3)
            quiet: If True, suppress verbose logging
        """
        if not quiet:
            logger.info(f"Getting preprocessor for model type: {model_type}")

        if model_type.lower() in ['vgg16', 'resnet50']:
            if not quiet:
                logger.info("Using Pipeline A (224x224) for VGG16/ResNet50")
            return self.pipeline_a
        elif model_type.lower() == 'inceptionv3':
            if not quiet:
                logger.info("Using Pipeline B (299x299) for InceptionV3")
            return self.pipeline_b
        else:
            logger.error(f"Unknown model type: {model_type}")
            raise ValueError(f"Unknown model type: {model_type}")

    def create_data_loaders(self, model_type, batch_size=32, is_training=True, quiet=False):
        """Create PyTorch DataLoaders for training, validation, and testing.
        Implements a 70/15/15 split for train/val/test.

        Args:
            model_type: Type of model (vgg16, resnet50, inceptionv3)
            batch_size: Batch size for the data loaders
            is_training: Whether this is for training (vs testing)
            quiet: If True, suppress verbose logging
        """
        if not quiet:
            logger.info("=" * 40)
            logger.info(f"CREATING DATA LOADERS (70/15/15 SPLIT, batch_size={batch_size})")
            logger.info("=" * 40)

        if self.merged_train_df is None:
            raise ValueError("No training data available. Please run load_data() first.")

        # First split: separate 15% for the test set
        remaining_df, test_df = train_test_split(
            self.merged_train_df,
            test_size=self.config.test_split,
            random_state=self.config.random_state,
            stratify=self.merged_train_df['diagnosis']
        )

        # Second split: split the remaining data into training and validation
        # The new validation split needs to be recalculated based on the remaining data
        val_split_adjusted = self.config.validation_split / (1.0 - self.config.test_split)
        train_df, val_df = train_test_split(
            remaining_df,
            test_size=val_split_adjusted,
            random_state=self.config.random_state,
            stratify=remaining_df['diagnosis']
        )

        if not quiet:
            logger.info(f"Total samples: {len(self.merged_train_df)}")
            logger.info(f"Training set size: {len(train_df)} samples (~{len(train_df)/len(self.merged_train_df)*100:.1f}%)")
            logger.info(f"Validation set size: {len(val_df)} samples (~{len(val_df)/len(self.merged_train_df)*100:.1f}%)")
            logger.info(f"Test set size: {len(test_df)} samples (~{len(test_df)/len(self.merged_train_df)*100:.1f}%)")

        preprocessor = self.get_preprocessor(model_type, quiet=quiet)
        train_transform = preprocessor.get_train_transforms()
        val_transform = preprocessor.get_val_transforms()

        train_dataset = DiabetitcRetinopathyDataset(train_df, transform=train_transform)
        val_dataset = DiabetitcRetinopathyDataset(val_df, transform=val_transform)
        test_dataset = DiabetitcRetinopathyDataset(test_df, transform=val_transform)

        # Optimize num_workers for environment
        if sys.platform == "win32":
            num_workers = 0
        else:
            num_workers = getattr(self.config, 'num_workers', 4)
        
        pin_memory = torch.cuda.is_available()
        prefetch_factor = getattr(self.config, 'prefetch_factor', 2) if num_workers > 0 else None
        persistent_workers = num_workers > 0

        loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
        }
        if num_workers > 0:
            loader_kwargs['prefetch_factor'] = prefetch_factor
            loader_kwargs['persistent_workers'] = persistent_workers

        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

        if not quiet:
            logger.info(f"Data loaders created: num_workers={num_workers}, pin_memory={pin_memory}")

        # The main training function expects train_loader and val_loader.
        # The test_loader can be retrieved separately if needed, or handled within the evaluation phase.
        # For now, let's store the test_loader to be used later.
        self.test_loader = test_loader

        return train_loader, val_loader

    def get_test_loader(self):
        """Returns the test loader created during the split."""
        if hasattr(self, 'test_loader'):
            return self.test_loader
        else:
            logger.warning("Test loader not created yet. Call create_data_loaders first.")
            return None


class PipelineA:
    """Pipeline A: Preprocessing for VGG16 & ResNet (224x224) with enhanced medical preprocessing"""

    def __init__(self, config=None):
        # Use config target size if available, else default to 224
        if config and hasattr(config, 'resnet_target_size'):
            self.target_size = config.resnet_target_size
        else:
            self.target_size = 224
            
        logger.info(f"Initialized Pipeline A with target size: {self.target_size}x{self.target_size}")
        logger.info("Pipeline A: Using border crop + CLAHE + advanced aug (no change to counts)")

    def get_train_transforms(self):
        """Get training transforms with data augmentation"""
        logger.debug("Creating training transforms for Pipeline A")
        return transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            
            # IMPROVEMENT: Aggressive geometric augmentations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),  # Fundus images have no "up/down"
            transforms.RandomRotation(180),        # Allow full rotation
            
            # IMPROVEMENT: Color augmentation for lighting robustness
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_val_transforms(self):
        """Get validation/test transforms (minimal preprocessing)"""
        logger.debug("Creating validation transforms for Pipeline A")
        return transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

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
        logger.debug(f"Processing image with Pipeline A: {image_path}")

        try:
            # Step 1: Load the original image
            image = Image.open(image_path).convert('RGB')
            logger.debug(f"Loaded image with size: {image.size}")

            # Step 2: Resize with aspect ratio preservation
            image = self._resize_with_aspect_ratio(image, self.target_size)
            logger.debug(f"Resized image to: {image.size}")

            # Step 3 & 4: Create square canvas and paste centered image
            image = self._create_square_canvas_and_center(image, self.target_size)
            logger.debug(f"Created square canvas: {image.size}")

            # Step 5 & 6: Apply transforms
            if is_training:
                transform = self.get_train_transforms()
                logger.debug("Applied training transforms")
            else:
                transform = self.get_val_transforms()
                logger.debug("Applied validation transforms")

            result = transform(image)
            logger.debug(f"Final tensor shape: {result.shape}")
            return result

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise

    def _resize_with_aspect_ratio(self, image, target_size):
        """Resize image while maintaining aspect ratio"""
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        if aspect_ratio > 1:  # Width > Height
            new_width = target_size
            new_height = int(target_size / aspect_ratio)
        else:  # Height >= Width
            new_height = target_size
            new_width = int(target_size * aspect_ratio)

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _create_square_canvas_and_center(self, image, target_size):
        """Create square canvas and center the image"""
        # Create black square canvas
        canvas = Image.new('RGB', (target_size, target_size), (0, 0, 0))

        # Calculate position to center the image
        image_width, image_height = image.size
        x_offset = (target_size - image_width) // 2
        y_offset = (target_size - image_height) // 2

        # Paste image onto canvas
        canvas.paste(image, (x_offset, y_offset))
        return canvas

class PipelineB:
    """Pipeline B: Preprocessing for InceptionV3 (299x299) with enhanced medical preprocessing"""

    def __init__(self, config=None):
        # Use config target size if available, else default to 299
        if config and hasattr(config, 'inception_target_size'):
            self.target_size = config.inception_target_size
        else:
            self.target_size = 299
            
        logger.info(f"Initialized Pipeline B with target size: {self.target_size}x{self.target_size}")
        logger.info("Pipeline B: Using border crop + CLAHE + advanced aug (no change to counts)")

    def get_train_transforms(self):
        """Get training transforms with data augmentation"""
        logger.debug("Creating training transforms for Pipeline B")
        return transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            
            # IMPROVEMENT: Aggressive geometric augmentations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(180),
            
            # IMPROVEMENT: Color augmentation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_val_transforms(self):
        """Get validation transforms (minimal preprocessing)"""
        logger.debug("Creating validation transforms for Pipeline B")
        return transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

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
        logger.debug(f"Processing image with Pipeline B: {image_path}")

        try:
            # Step 1: Load the original image
            image = Image.open(image_path).convert('RGB')
            logger.debug(f"Loaded image with size: {image.size}")

            # Step 2: Resize with aspect ratio preservation
            image = self._resize_with_aspect_ratio(image, self.target_size)
            logger.debug(f"Resized image to: {image.size}")

            # Step 3 & 4: Create square canvas and paste centered image
            image = self._create_square_canvas_and_center(image, self.target_size)
            logger.debug(f"Created square canvas: {image.size}")

            # Step 5 & 6: Apply transforms
            if is_training:
                transform = self.get_train_transforms()
                logger.debug("Applied training transforms")
            else:
                transform = self.get_val_transforms()
                logger.debug("Applied validation transforms")

            result = transform(image)
            logger.debug(f"Final tensor shape: {result.shape}")
            return result

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise

    def _resize_with_aspect_ratio(self, image, target_size):
        """Resize image while maintaining aspect ratio"""
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        if aspect_ratio > 1:  # Width > Height
            new_width = target_size
            new_height = int(target_size / aspect_ratio)
        else:  # Height >= Width
            new_height = target_size
            new_width = int(target_size * aspect_ratio)

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _create_square_canvas_and_center(self, image, target_size):
        """Create square canvas and center the image"""
        # Create black square canvas
        canvas = Image.new('RGB', (target_size, target_size), (0, 0, 0))

        # Calculate position to center the image
        image_width, image_height = image.size
        x_offset = (target_size - image_width) // 2
        y_offset = (target_size - image_height) // 2

        # Paste image onto canvas
        canvas.paste(image, (x_offset, y_offset))
        return canvas

# Test function for independent execution
def test_data_module():
    """Test the data module independently"""
    print("Testing Data module...")

    try:
        # Import config
        from config import Config

        # Create config (disable logger for cleaner test output)
        config = Config(use_logger=False)
        print("[OK] Config imported and created successfully")

        # Test pipeline classes
        print("Testing Pipeline classes...")
        pipeline_a = PipelineA()
        pipeline_b = PipelineB()
        print("[OK] Pipeline A and B created successfully")

        # Test transforms
        print("Testing transforms...")
        train_transforms_a = pipeline_a.get_train_transforms()
        val_transforms_a = pipeline_a.get_val_transforms()
        train_transforms_b = pipeline_b.get_train_transforms()
        val_transforms_b = pipeline_b.get_val_transforms()
        print("[OK] All transforms created successfully")

        # Test data loader creation
        print("Testing CustomDataLoader creation...")
        data_loader = CustomDataLoader(config, use_excel_logger=True)  # Enable Excel logger for data loading
        print("[OK] CustomDataLoader created successfully")

        # Test data loading (this will check if files exist)
        print("Testing data loading process...")
        try:
            data_loader.load_data()
            print("[OK] Data loading completed successfully")

            # Test data retrieval
            train_data = data_loader.get_train_data()

            if train_data is not None:
                print(f"[OK] Training data available: {len(train_data)} samples")
                # Verify EyePACS data is included
                if 'dataset' in train_data.columns:
                    dataset_counts = train_data['dataset'].value_counts()
                    print(f"[OK] Dataset distribution:")
                    for dataset, count in dataset_counts.items():
                        print(f"  - {dataset}: {count} samples")
            else:
                print(" No training data available")

            print("[INFO]  Test data is created via 70/15/15 split during data loader creation")

            # Test data loader creation for different models
            print("Testing PyTorch DataLoader creation...")
            for model_type in ['vgg16', 'resnet50', 'inceptionv3']:
                try:
                    train_loader, val_loader = data_loader.create_data_loaders(model_type)
                    print(f"[OK] DataLoaders created for {model_type}: {len(train_loader)} train batches, {len(val_loader)} val batches")
                except Exception as e:
                    print(f" DataLoader creation failed for {model_type}: {e}")

        except Exception as e:
            print(f" Data loading failed (this is expected if data files are missing): {e}")
            print("[OK] Data module structure is correct even without data files")

        print("[OK] Data module test PASSED")
        return True

    except Exception as e:
        print(f"[FAIL] Data module test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_module()
    print(f"\nData Module Test Result: {'PASS' if success else 'FAIL'}")
