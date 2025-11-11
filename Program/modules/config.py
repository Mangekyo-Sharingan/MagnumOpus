"""
Configuration module for diabetic retinopathy classification project
"""
import os
import sys
from pathlib import Path

# Add utils to path for Excel logger
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from excel_logger import ExcelLogger

class Config:
    """Configuration class containing all project settings"""

    def __init__(self, use_logger=True):
        self.use_logger = use_logger
        self.logger = None

        if self.use_logger:
            self.logger = ExcelLogger("config")
            self.logger.__enter__()
            self.logger.log("Initializing Config module...")

        # Project paths
        self.base_dir = Path(__file__).parent.parent.parent
        self.data_dir = self.base_dir / "Data"

        # APTOS dataset paths
        self.aptos_dir = self.data_dir / "Aptos"
        self.aptos_train_images_dir = self.aptos_dir / "train_images"
        self.aptos_test_images_dir = self.aptos_dir / "test_images"
        self.aptos_train_csv = self.aptos_dir / "train.csv"
        self.aptos_test_csv = self.aptos_dir / "test.csv"

        # EyePACS dataset paths
        self.eyepacs_dir = self.data_dir / "EyePacs"
        self.eyepacs_train_images_dir = self.eyepacs_dir / "train"
        self.eyepacs_test_images_dir = self.eyepacs_dir / "test"
        self.eyepacs_train_csv = self.eyepacs_dir / "trainLabels.csv" / "trainLabels.csv"  # Fixed: CSV is inside a directory

        # Training parameters
        self.batch_size = 32
        self.validation_split = 0.2
        self.random_state = 20020315
        self.num_classes = 5  # 0-4 severity levels
        self.epochs = 50
        self.learning_rate = 0.005

        # Model-specific parameters
        self.model_configs = {
            'vgg16': {
                'image_size': (224, 224),
                'pipeline': 'A'
            },
            'resnet50': {
                'image_size': (224, 224),
                'pipeline': 'A'
            },
            'inceptionv3': {
                'image_size': (299, 299),
                'pipeline': 'B'
            }
        }

        # Available models
        self.models = ['vgg16', 'resnet50', 'inceptionv3']

        if self.use_logger:
            self.logger.log("Configuration initialized successfully")
            self.logger.log_separator()
            self.logger.log("Path Configuration:")
            self.logger.log_dict({
                'base_dir': str(self.base_dir),
                'data_dir': str(self.data_dir),
                'aptos_dir': str(self.aptos_dir),
                'eyepacs_dir': str(self.eyepacs_dir)
            })
            self.logger.log_separator()
            self.logger.log("Training Parameters:")
            self.logger.log_dict({
                'batch_size': self.batch_size,
                'validation_split': self.validation_split,
                'random_state': self.random_state,
                'num_classes': self.num_classes,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate
            })
            self.logger.log_separator()
            self.logger.log(f"Available models: {', '.join(self.models)}")

    def __del__(self):
        """Destructor to close logger"""
        if self.use_logger and self.logger:
            try:
                self.logger.__exit__(None, None, None)
            except:
                pass

    def print_config(self):
        """Print configuration for debugging"""
        if self.use_logger:
            self.logger.log("Printing configuration...")

        print("=" * 50)
        print("CONFIGURATION SETTINGS")
        print("=" * 50)
        print(f"Base directory: {self.base_dir}")
        print(f"Data directory: {self.data_dir}")
        print(f"APTOS train CSV: {self.aptos_train_csv}")
        print(f"APTOS test CSV: {self.aptos_test_csv}")
        print(f"EyePACS train CSV: {self.eyepacs_train_csv}")
        print(f"Batch size: {self.batch_size}")
        print(f"Validation split: {self.validation_split}")
        print(f"Random state: {self.random_state}")
        print(f"Available models: {self.models}")
        print("=" * 50)

        if self.use_logger:
            self.logger.log("Configuration printed to console")

    def create_directories(self):
        """Create necessary directories for the project"""
        if self.use_logger:
            self.logger.log("Creating project directories...")

        directories = [
            self.base_dir / "models",
            self.base_dir / "results",
            self.base_dir / "logs"
        ]

        created_count = 0
        for directory in directories:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                created_count += 1
                if self.use_logger:
                    self.logger.log(f"Created directory: {directory}")
            else:
                if self.use_logger:
                    self.logger.log(f"Directory already exists: {directory}")

        if self.use_logger:
            self.logger.log(f"Directory creation complete. Created {created_count} new directories.")

        print(f"✓ Directories created/verified: {len(directories)}")

# Test function for independent execution
def test_config():
    """Test the configuration module independently"""
    print("Testing Config module...")

    try:
        config = Config()
        print("✓ Config object created successfully")

        # Test path existence
        print(f"✓ Base directory exists: {config.base_dir.exists()}")
        print(f"✓ Data directory exists: {config.data_dir.exists()}")
        print(f"✓ APTOS directory exists: {config.aptos_dir.exists()}")
        print(f"✓ EyePACS directory exists: {config.eyepacs_dir.exists()}")

        # Print configuration
        config.print_config()

        print("✓ Config module test PASSED")
        return True

    except Exception as e:
        print(f"✗ Config module test FAILED: {e}")
        return False

if __name__ == "__main__":
    success = test_config()
    print(f"\nConfig Module Test Result: {'PASS' if success else 'FAIL'}")
