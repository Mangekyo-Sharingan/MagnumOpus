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
        self.eyepacs_train_csv = self.eyepacs_dir / "trainLabels.csv" / "trainLabels.csv"

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

        # General model parameters
        self.batch_size = 32
        self.num_classes = 5  # 0-4 severity levels
        self.epochs = 50
        self.learning_rate = 0.001

        # Available models
        self.models = ['vgg16', 'resnet50', 'inceptionv3']

        # Training parameters
        self.validation_split = 0.2
        self.random_state = 42

        # Data augmentation parameters for training
        self.augmentation_params = {
            'rotation_range': 20,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'horizontal_flip': True,
            'zoom_range': 0.2,
            'fill_mode': 'constant',
            'cval': 0  # Fill with black
        }

    def create_directories(self):
        """Create necessary output directories"""
        output_dirs = [
            self.base_dir / "models",
            self.base_dir / "results",
            self.base_dir / "logs"
        ]
        for directory in output_dirs:
            directory.mkdir(parents=True, exist_ok=True)

    def get_model_config(self, model_name):
        """Get specific configuration for a model"""
        if model_name.lower() not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        return self.model_configs[model_name.lower()]

    def get_image_size(self, model_name):
        """Get image size for specific model"""
        return self.get_model_config(model_name)['image_size']

    def get_pipeline_type(self, model_name):
        """Get pipeline type (A or B) for specific model"""
        return self.get_model_config(model_name)['pipeline']
