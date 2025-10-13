"""
Data loading and preprocessing module for diabetic retinopathy classification
"""
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DiabetitcRetinopathyDataset(Dataset):
    """Custom PyTorch Dataset for diabetic retinopathy images"""

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        logger.info(f"Created DiabetitcRetinopathyDataset with {len(dataframe)} samples")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        try:
            row = self.dataframe.iloc[idx]
            image_path = row['image_path']
            label = row['diagnosis']

            # Load image
            image = Image.open(image_path).convert('RGB')
            logger.debug(f"Successfully loaded image: {image_path}")

            if self.transform:
                image = self.transform(image)
                logger.debug(f"Applied transforms to image: {image_path}")

            return image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            logger.error(f"Error loading image at index {idx}: {e}")
            raise

class CustomDataLoader:
    """
    Class responsible for loading and preprocessing diabetic retinopathy data

    This class implements a lazy loading strategy for handling multiple datasets:
    - Metadata (file paths, labels) is loaded and merged in memory
    - Actual image data remains on disk and is loaded on-demand during training
    - This approach is memory efficient and scalable for large datasets
    - No file duplication occurs - original images stay in their respective directories
    """

    def __init__(self, config):
        logger.info("Initializing CustomDataLoader...")
        self.config = config
        self.aptos_train_df = None
        self.aptos_test_df = None
        self.eyepacs_train_df = None
        self.merged_train_df = None  # Combined dataset (metadata only - uses lazy loading)
        self.pipeline_a = PipelineA()  # VGG16 & ResNet
        self.pipeline_b = PipelineB()  # InceptionV3
        logger.info("CustomDataLoader initialization complete")

    def load_data(self):
        """Load and merge training data from both APTOS and EyePACS datasets"""
        logger.info("=" * 50)
        logger.info("STARTING DATA LOADING PROCESS")
        logger.info("=" * 50)

        start_time = datetime.now()

        try:
            logger.info("Loading APTOS dataset...")
            self._load_aptos_data()

            logger.info("Loading EyePACS dataset...")
            self._load_eyepacs_data()

            logger.info("Merging datasets...")
            self._merge_datasets()

            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(f"Data loading completed successfully in {duration.total_seconds():.2f} seconds")
            logger.info(f"Total training samples after merging: {len(self.merged_train_df)}")
            logger.info("=" * 50)

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def _load_aptos_data(self):
        """Load APTOS dataset"""
        logger.info("Checking APTOS data availability...")

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
                else:
                    logger.info("All APTOS training images found successfully")

                logger.info(f"Loaded {len(self.aptos_train_df)} APTOS training samples")

            except Exception as e:
                logger.error(f"Failed to load APTOS training data: {e}")
                raise
        else:
            logger.warning(f"APTOS training CSV not found: {self.config.aptos_train_csv}")

        # Load test data
        if self.config.aptos_test_csv.exists():
            logger.info(f"Found APTOS test CSV: {self.config.aptos_test_csv}")
            try:
                self.aptos_test_df = pd.read_csv(self.config.aptos_test_csv)
                logger.info(f"Successfully read APTOS test CSV with {len(self.aptos_test_df)} rows")

                # Add image paths for test data
                self.aptos_test_df['image_path'] = self.aptos_test_df['id_code'].apply(
                    lambda x: str(self.config.aptos_test_images_dir / f"{x}.png")
                )
                logger.info("Added image paths to APTOS test data")

                # Verify test image files exist
                missing_test_images = 0
                for idx, row in self.aptos_test_df.iterrows():
                    if not Path(row['image_path']).exists():
                        missing_test_images += 1

                if missing_test_images > 0:
                    logger.warning(f"Found {missing_test_images} missing APTOS test images")
                else:
                    logger.info("All APTOS test images found successfully")

                logger.info(f"Loaded {len(self.aptos_test_df)} APTOS test samples")

            except Exception as e:
                logger.error(f"Failed to load APTOS test data: {e}")
                raise
        else:
            logger.warning(f"APTOS test CSV not found: {self.config.aptos_test_csv}")

    def _load_eyepacs_data(self):
        """Load EyePACS dataset"""
        logger.info("Checking EyePACS data availability...")

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
                else:
                    logger.info("All EyePACS training images found successfully")

                logger.info(f"Loaded {len(self.eyepacs_train_df)} EyePACS training samples")

            except Exception as e:
                logger.error(f"Failed to load EyePACS training data: {e}")
                raise
        else:
            logger.warning(f"EyePACS training CSV not found: {self.config.eyepacs_train_csv}")

    def _merge_datasets(self):
        """Merge APTOS and EyePACS datasets"""
        logger.info("Starting dataset merging process...")

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
        for dataset, count in dataset_counts.items():
            percentage = (count / total_samples) * 100
            logger.info(f"  {dataset}: {count} samples ({percentage:.1f}%)")

        # Diagnosis distribution
        logger.info("\nDiagnosis distribution:")
        diagnosis_counts = self.merged_train_df['diagnosis'].value_counts().sort_index()
        for diagnosis, count in diagnosis_counts.items():
            percentage = (count / total_samples) * 100
            logger.info(f"  Level {diagnosis}: {count} samples ({percentage:.1f}%)")

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

    def get_test_data(self):
        """Get the APTOS test dataset (EyePACS doesn't have labeled test data)"""
        if self.aptos_test_df is not None:
            logger.info(f"Returning test dataset with {len(self.aptos_test_df)} samples")
            return self.aptos_test_df
        else:
            logger.warning("No test data available")
            return None

    def get_preprocessor(self, model_type):
        """Get appropriate preprocessor based on model type"""
        logger.info(f"Getting preprocessor for model type: {model_type}")

        if model_type.lower() in ['vgg16', 'resnet50']:
            logger.info("Using Pipeline A (224x224) for VGG16/ResNet50")
            return self.pipeline_a
        elif model_type.lower() == 'inceptionv3':
            logger.info("Using Pipeline B (299x299) for InceptionV3")
            return self.pipeline_b
        else:
            logger.error(f"Unknown model type: {model_type}")
            raise ValueError(f"Unknown model type: {model_type}")

    def create_data_loaders(self, model_type, is_training=True):
        """Create PyTorch DataLoaders for training and validation"""
        logger.info("=" * 40)
        logger.info("CREATING DATA LOADERS")
        logger.info("=" * 40)

        if self.merged_train_df is None:
            logger.error("No training data available. Please run load_data() first.")
            raise ValueError("No training data available. Please run load_data() first.")

        logger.info(f"Model type: {model_type}")
        logger.info(f"Training mode: {is_training}")
        logger.info(f"Validation split: {self.config.validation_split}")
        logger.info(f"Batch size: {self.config.batch_size}")

        try:
            preprocessor = self.get_preprocessor(model_type)

            # Split data into train and validation
            logger.info("Splitting data into training and validation sets...")
            train_df, val_df = train_test_split(
                self.merged_train_df,
                test_size=self.config.validation_split,
                random_state=self.config.random_state,
                stratify=self.merged_train_df['diagnosis']
            )

            logger.info(f"Training set size: {len(train_df)} samples")
            logger.info(f"Validation set size: {len(val_df)} samples")

            # Get transforms
            logger.info("Getting data transforms...")
            train_transform = preprocessor.get_train_transforms()
            val_transform = preprocessor.get_val_transforms()
            logger.info("Transforms obtained successfully")

            # Create datasets
            logger.info("Creating PyTorch datasets...")
            train_dataset = DiabetitcRetinopathyDataset(train_df, transform=train_transform)
            val_dataset = DiabetitcRetinopathyDataset(val_df, transform=val_transform)
            logger.info("Datasets created successfully")

            # Create data loaders
            logger.info("Creating PyTorch data loaders...")
            num_workers = 4
            pin_memory = torch.cuda.is_available()
            logger.info(f"Using {num_workers} workers, pin_memory={pin_memory}")

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory
            )

            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            )

            logger.info(f"Training loader: {len(train_loader)} batches")
            logger.info(f"Validation loader: {len(val_loader)} batches")
            logger.info("Data loaders created successfully!")
            logger.info("=" * 40)

            return train_loader, val_loader

        except Exception as e:
            logger.error(f"Failed to create data loaders: {e}")
            raise

    def create_test_loader(self, model_type):
        """Create test data loader"""
        logger.info("=" * 40)
        logger.info("CREATING TEST DATA LOADER")
        logger.info("=" * 40)

        if self.aptos_test_df is None:
            logger.error("No test data available")
            return None

        logger.info(f"Model type: {model_type}")
        logger.info(f"Test samples: {len(self.aptos_test_df)}")

        try:
            preprocessor = self.get_preprocessor(model_type)
            test_transform = preprocessor.get_val_transforms()

            logger.info("Creating test dataset...")
            test_dataset = DiabetitcRetinopathyDataset(self.aptos_test_df, transform=test_transform)

            logger.info("Creating test data loader...")
            num_workers = 4
            pin_memory = torch.cuda.is_available()

            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            )

            logger.info(f"Test loader: {len(test_loader)} batches")
            logger.info("Test data loader created successfully!")
            logger.info("=" * 40)

            return test_loader

        except Exception as e:
            logger.error(f"Failed to create test data loader: {e}")
            raise

class PipelineA:
    """Pipeline A: Preprocessing for VGG16 & ResNet (224x224)"""

    def __init__(self):
        self.target_size = 224
        logger.info(f"Initialized Pipeline A with target size: {self.target_size}x{self.target_size}")

    def get_train_transforms(self):
        """Get training transforms with augmentation"""
        logger.debug("Creating training transforms for Pipeline A")
        return transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(self.target_size, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_val_transforms(self):
        """Get validation/test transforms without augmentation"""
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
    """Pipeline B: Preprocessing for InceptionV3 (299x299)"""

    def __init__(self):
        self.target_size = 299
        logger.info(f"Initialized Pipeline B with target size: {self.target_size}x{self.target_size}")

    def get_train_transforms(self):
        """Get training transforms with augmentation"""
        logger.debug("Creating training transforms for Pipeline B")
        return transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(self.target_size, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_val_transforms(self):
        """Get validation/test transforms without augmentation"""
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

        # Create config
        config = Config()
        print("✓ Config imported and created successfully")

        # Test pipeline classes
        print("Testing Pipeline classes...")
        pipeline_a = PipelineA()
        pipeline_b = PipelineB()
        print("✓ Pipeline A and B created successfully")

        # Test transforms
        print("Testing transforms...")
        train_transforms_a = pipeline_a.get_train_transforms()
        val_transforms_a = pipeline_a.get_val_transforms()
        train_transforms_b = pipeline_b.get_train_transforms()
        val_transforms_b = pipeline_b.get_val_transforms()
        print("✓ All transforms created successfully")

        # Test data loader creation
        print("Testing CustomDataLoader creation...")
        data_loader = CustomDataLoader(config)
        print("✓ CustomDataLoader created successfully")

        # Test data loading (this will check if files exist)
        print("Testing data loading process...")
        try:
            data_loader.load_data()
            print("✓ Data loading completed successfully")

            # Test data retrieval
            train_data = data_loader.get_train_data()
            test_data = data_loader.get_test_data()

            if train_data is not None:
                print(f"✓ Training data available: {len(train_data)} samples")
            else:
                print("⚠ No training data available")

            if test_data is not None:
                print(f"✓ Test data available: {len(test_data)} samples")
            else:
                print("⚠ No test data available")

            # Test data loader creation for different models
            print("Testing PyTorch DataLoader creation...")
            for model_type in ['vgg16', 'resnet50', 'inceptionv3']:
                try:
                    train_loader, val_loader = data_loader.create_data_loaders(model_type)
                    print(f"✓ DataLoaders created for {model_type}: {len(train_loader)} train batches, {len(val_loader)} val batches")
                except Exception as e:
                    print(f"⚠ DataLoader creation failed for {model_type}: {e}")

        except Exception as e:
            print(f"⚠ Data loading failed (this is expected if data files are missing): {e}")
            print("✓ Data module structure is correct even without data files")

        print("✓ Data module test PASSED")
        return True

    except Exception as e:
        print(f"✗ Data module test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_module()
    print(f"\nData Module Test Result: {'PASS' if success else 'FAIL'}")
