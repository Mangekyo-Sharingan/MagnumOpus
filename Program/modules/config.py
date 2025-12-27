#Configuration module for diabetic retinopathy classification project
import os
import sys
from pathlib import Path

# Add utils to path for Excel logger
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

try:
    from excel_logger import ExcelLogger
except ImportError:
    from ..utils.excel_logger import ExcelLogger

class Config:
    #Configuration class containing all project settings

    @staticmethod
    def _env_flag(var_name, default=True):
        value = os.getenv(var_name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}


    @staticmethod
    def _get_env_int(var_name, default):
        """Get integer from environment variable."""
        value = os.getenv(var_name)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    @staticmethod
    def _get_env_float(var_name, default):
        """Get float from environment variable."""
        value = os.getenv(var_name)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default

    @staticmethod
    def _get_env_list(var_name, default):
        """Get list from comma-separated environment variable."""
        value = os.getenv(var_name)
        if value is None:
            return default
        return [x.strip() for x in value.split(",") if x.strip()]

    def __init__(self, use_logger=True):
        # Batch mode configuration
        self.batch_mode = self._env_flag("BATCH_MODE", False)

        # Logger configuration
        self.use_logger = use_logger
        self.logger = None

        if self.use_logger:
            self.logger = ExcelLogger("config")
            self.logger.__enter__()
            self.logger.log("Initializing Config module...")

        # Project paths - allow override via environment
        self.base_dir = Path(os.environ.get("PROJECT_BASE_DIR", Path(__file__).parent.parent.parent))
        self.data_dir = Path(os.environ.get("DATA_DIR", self.base_dir / "Data"))

        # APTOS dataset paths
        self.aptos_dir = self.data_dir / "Aptos"
        self.aptos_train_images_dir = self.aptos_dir / "train_images"
        self.aptos_train_csv = self.aptos_dir / "train.csv"

        # EyePACS dataset paths
        self.eyepacs_dir = self.data_dir / "EyePacs"
        self.eyepacs_train_images_dir = self.eyepacs_dir / "train"
        self.eyepacs_train_csv = self.eyepacs_dir / "trainLabels.csv" / "trainLabels.csv"  # Fixed: CSV is inside a directory

        # Training parameters - allow override via environment for Ray jobs
        self.validation_split = self._get_env_float("VALIDATION_SPLIT", 0.15)
        self.test_split = self._get_env_float("TEST_SPLIT", 0.15)
        self.random_state = self._get_env_int("RANDOM_STATE", 20020315)

        # Model Input Resolutions (V2 Enhanced)
        self.resnet_target_size = self._get_env_int("RESNET_TARGET_SIZE", 256)  # V2: increased from 224
        self.inception_target_size = self._get_env_int("INCEPTION_TARGET_SIZE", 299)  # V2: native resolution
        self.num_classes = 5
        self.class_names = {
            0: "No DR",
            1: "Mild",
            2: "Moderate",
            3: "Severe",
            4: "Proliferative"
        }

        # Mixed Precision Training (FP16) for better GPU utilization
        self.use_amp = self._env_flag("USE_AMP", False)

        # Model-specific parameters including hyperparameters (with env overrides)
        default_batch_size = self._get_env_int("BATCH_SIZE", 64)
        default_epochs = self._get_env_int("NUM_EPOCHS", 50)
        default_lr = self._get_env_float("LEARNING_RATE", 0.001)
        
        # Enhanced V2 image sizes: ResNet50 uses 256x256, InceptionV3 uses 299x299
        self.model_configs = {
            'vgg16': {
                'image_size': (224, 224),  # VGG16 native resolution
                'pipeline': 'A',
                'batch_size': self._get_env_int("VGG16_BATCH_SIZE", default_batch_size),
                'epochs': self._get_env_int("VGG16_EPOCHS", default_epochs),
                'learning_rate': self._get_env_float("VGG16_LR", default_lr)
            },
            'resnet50': {
                'image_size': (256, 256),  # V2 Enhanced: increased from 224x224 for better lesion detection
                'pipeline': 'A',
                'batch_size': self._get_env_int("RESNET50_BATCH_SIZE", default_batch_size),
                'epochs': self._get_env_int("RESNET50_EPOCHS", default_epochs),
                'learning_rate': self._get_env_float("RESNET50_LR", default_lr)
            },
            'inceptionv3': {
                'image_size': (299, 299),  # V2 Enhanced: native InceptionV3 resolution
                'pipeline': 'B',
                'batch_size': self._get_env_int("INCEPTIONV3_BATCH_SIZE", default_batch_size),
                'epochs': self._get_env_int("INCEPTIONV3_EPOCHS", default_epochs),
                'learning_rate': self._get_env_float("INCEPTIONV3_LR", default_lr)
            }
        }

        # Available models - can be overridden via environment
        self.models = self._get_env_list("MODELS", ['vgg16', 'resnet50', 'inceptionv3'])

        # Data loading settings
        self.num_workers = self._get_env_int("NUM_DATA_WORKERS", 14)
        self.prefetch_factor = self._get_env_int("PREFETCH_FACTOR", 2)
        
        # MIOpen fix toggles
        self.enable_miopen_fix = self._env_flag("ENABLE_MIOPEN_FIX", True)
        self.miopen_fix_sources = {
            'device_manager': self._env_flag("MIOPEN_FIX_DEVICE_MANAGER", True),
            'fix_miopen_amd': self._env_flag("MIOPEN_FIX_AMD_SCRIPT", False),
            'fix_miopen_error': self._env_flag("MIOPEN_FIX_LEGACY_SCRIPT", False)
        }

        # Performance optimization
        self.miopen_disable_cache = self._env_flag("MIOPEN_DISABLE_CACHE_OPT", False)

        # Transfer Learning Configuration
        self.use_transfer_learning = True
        self.stage1_percentage = 0.4
        self.stage2_lr_multiplier = 0.1

        # Model-specific unfreezing configuration
        self.unfreeze_layers = {
            'vgg16': 'features.24',
            'resnet50': 'layer4',
            'inceptionv3': 'Mixed_7a'
        }

        # High-impact training improvements
        self.use_label_smoothing = True
        self.label_smoothing = 0.15  # Increased from 0.1 to reduce overfitting
        self.use_focal_loss = True
        self.focal_gamma = 2.0
        self.focal_alpha = None

        # Data mixing augmentations (increased to combat overfitting)
        self.use_mixup = True
        self.mixup_alpha = 0.4  # Increased from 0.2 for stronger regularization
        self.use_cutmix = True
        self.cutmix_alpha = 1.0
        self.mix_prob = 0.5  # Increased from 0.5 to apply more often

        # Early stopping to prevent overfitting
        self.use_early_stopping = True
        self.early_stopping_patience = 15  # Stop if val loss doesn't improve for 10 epochs
        self.early_stopping_min_delta = 0.00075  # Minimum change to qualify as improvement

        # Gradient clipping to stabilize training
        self.use_gradient_clipping = True
        self.gradient_clip_value = 1.0

        # Stronger weight decay for regularization
        self.weight_decay = 2e-4  # Increased from 1e-4

        # Test-Time Augmentation
        self.use_tta = True
        self.tta_flip = True

        if self.use_logger:
            self.logger.log("Configuration initialized successfully")
            self.logger.__exit__(None, None, None)

    def get_model_config(self, model_name):
        """Returns the configuration for a specific model."""
        if model_name not in self.model_configs:
            raise ValueError(f"Configuration for model '{model_name}' not found.")
        return self.model_configs[model_name]

    def __del__(self):
        """Destructor to close logger"""
        if self.use_logger and self.logger:
            try:
                self.logger.__exit__(None, None, None)
            except:
                pass

    def print_config(self):
        """Print configuration for debugging"""
        print("=" * 50)
        print("CONFIGURATION SETTINGS")
        print("=" * 50)
        print(f"Base directory: {self.base_dir}")
        print(f"Data directory: {self.data_dir}")
        print(f"APTOS train CSV: {self.aptos_train_csv}")
        print(f"EyePACS train CSV: {self.eyepacs_train_csv}")
        print(f"Validation split: {self.validation_split}")
        print(f"Test split: {self.test_split}")
        print(f"Random state: {self.random_state}")
        print(f"Available models: {self.models}")
        print(f"Enable MIOpen fix: {self.enable_miopen_fix}")
        print("=" * 50)


    def create_directories(self):
        """Create necessary directories for the project"""
        directories = [
            self.base_dir / "models",
            self.base_dir / "results",
            self.base_dir / "logs"
        ]

        for directory in directories:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)

        print(f"[OK] Directories created/verified: {len(directories)}")

# Test function for independent execution
def test_config():
    """Test the configuration module independently"""
    print("Testing Config module...")

    try:
        config = Config()
        print("[OK] Config object created successfully")

        # Test path existence
        print(f"[OK] Base directory exists: {config.base_dir.exists()}")
        print(f"[OK] Data directory exists: {config.data_dir.exists()}")
        print(f"[OK] APTOS directory exists: {config.aptos_dir.exists()}")
        print(f"[OK] EyePACS directory exists: {config.eyepacs_dir.exists()}")

        # Print configuration
        config.print_config()

        print("[OK] Config module test PASSED")
        return True

    except Exception as e:
        print(f"[FAIL] Config module test FAILED: {e}")
        return False

if __name__ == "__main__":
    success = test_config()
    print(f"\nConfig Module Test Result: {'PASS' if success else 'FAIL'}")
