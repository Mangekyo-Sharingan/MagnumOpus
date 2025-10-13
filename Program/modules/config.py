"""
Configuration module for diabetic retinopathy classification project
"""
import os
from pathlib import Path

class Config:
    """Configuration class containing all project settings"""

    def __init__(self):
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
        self.eyepacs_train_csv = self.eyepacs_dir / "trainLabels.csv"

        # Training parameters
        self.batch_size = 32
        self.validation_split = 0.2
        self.random_state = 42
        self.num_classes = 5  # 0-4 severity levels
        self.epochs = 50
        self.learning_rate = 0.001

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

    def print_config(self):
        """Print configuration for debugging"""
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
