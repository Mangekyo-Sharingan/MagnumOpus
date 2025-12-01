"""
Utility functions for diabetic retinopathy classification project
"""
import os
import json
import pickle
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import subprocess
import sys

class Utils:
    """Utility class containing helper functions"""

    @staticmethod
    def create_directory(directory_path):
        """Create directory if it doesn't exist"""
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        print(f"Directory created/verified: {directory_path}")

    @staticmethod
    def save_json(data, filepath):
        """Save data to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"JSON data saved to: {filepath}")

    @staticmethod
    def load_json(filepath):
        """Load data from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"JSON data loaded from: {filepath}")
        return data

    @staticmethod
    def save_pickle(obj, filepath):
        """Save object to pickle file"""
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        print(f"Pickle object saved to: {filepath}")

    @staticmethod
    def load_pickle(filepath):
        """Load object from pickle file"""
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        print(f"Pickle object loaded from: {filepath}")
        return obj

    @staticmethod
    def get_timestamp():
        """Get current timestamp string"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def print_system_info():
        """Print system information"""
        print("=" * 50)
        print("SYSTEM INFORMATION")
        print("=" * 50)
        print(f"Python version: {os.sys.version}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Current working directory: {os.getcwd()}")
        print("=" * 50)

    @staticmethod
    def set_seed(seed=2002):
        """Set random seed for reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed set to: {seed}")

    @staticmethod
    def count_parameters(model):
        """Count total and trainable parameters in a model"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }

    @staticmethod
    def format_time(seconds):
        """Format time in seconds to human readable format"""
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = seconds // 60
            seconds = seconds % 60
            return f"{int(minutes)}m {seconds:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            seconds = seconds % 60
            return f"{int(hours)}h {int(minutes)}m {seconds:.0f}s"

    @staticmethod
    def create_experiment_dir(base_dir, experiment_name=None):
        """Create experiment directory with timestamp"""
        if experiment_name is None:
            experiment_name = f"experiment_{Utils.get_timestamp()}"

        exp_dir = Path(base_dir) / experiment_name
        Utils.create_directory(exp_dir)

        # Create subdirectories
        Utils.create_directory(exp_dir / "models")
        Utils.create_directory(exp_dir / "plots")
        Utils.create_directory(exp_dir / "logs")
        Utils.create_directory(exp_dir / "results")

        return exp_dir

    @staticmethod
    def plot_metrics_comparison(metrics_dict, save_path=None):
        """Plot comparison of metrics across different models"""
        models = list(metrics_dict.keys())
        metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        for i, metric in enumerate(metrics):
            values = [metrics_dict[model].get(metric, 0) for model in models]
            axes[i].bar(models, values, alpha=0.7)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            axes[i].set_ylim(0, 1)

            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Metrics comparison plot saved to: {save_path}")

        plt.show()

    @staticmethod
    def log_experiment(exp_dir, config, model_name, metrics, notes=""):
        """Log experiment details"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'config': {
                'batch_size': config.batch_size,
                'epochs': config.epochs,
                'learning_rate': config.learning_rate,
                'validation_split': config.validation_split,
                'random_state': config.random_state
            },
            'metrics': metrics,
            'notes': notes
        }

        log_file = exp_dir / "logs" / f"{model_name}_experiment_log.json"
        Utils.save_json(log_data, log_file)

        return log_file

    @staticmethod
    def find_previous_hyperparams(model_name: str, base_dir: Path) -> dict | None:
        """Locate previous hyperparameters for a model from saved model packages.
        Prefer README.md (as requested) and fall back to metadata JSON if needed.

        Args:
            model_name: One of {'vgg16','resnet50','inceptionv3'}
            base_dir: Project root Path
        Returns:
            dict with keys {'batch_size','learning_rate','epochs','source_dir'} or None
        """
        models_root = base_dir / "models"
        if not models_root.exists():
            return None

        # Collect candidate package dirs that start with model_name
        candidates: list[Path] = [p for p in models_root.iterdir() if p.is_dir() and p.name.lower().startswith(model_name.lower() + "_")]
        if not candidates:
            return None

        def read_metadata(d: Path) -> tuple[dict | None, str]:
            for meta in d.glob("*_metadata.json"):
                try:
                    data = json.load(open(meta, "r", encoding="utf-8"))
                    params = {
                        "batch_size": int(data.get("batch_size")) if data.get("batch_size") is not None else None,
                        "learning_rate": float(data.get("learning_rate")) if data.get("learning_rate") is not None else None,
                        "epochs": int(data.get("total_epochs_trained")) if data.get("total_epochs_trained") is not None else None,
                    }
                    date_key = str(data.get("training_date") or "")
                    return params, date_key
                except Exception:
                    continue
            return None, ""

        def parse_readme(d: Path) -> tuple[dict | None, str]:
            rd = d / "README.md"
            if not rd.exists():
                return None, ""
            try:
                text = rd.read_text(encoding="utf-8", errors="ignore")
                bs = re.search(r"Batch Size\s*:\s*(\d+)", text, re.IGNORECASE)
                lr = re.search(r"Learning Rate\s*:\s*([0-9.eE-]+)", text, re.IGNORECASE)
                ep = re.search(r"Total Epochs\s*:\s*(\d+)", text, re.IGNORECASE)
                params = {
                    "batch_size": int(bs.group(1)) if bs else None,
                    "learning_rate": float(lr.group(1)) if lr else None,
                    "epochs": int(ep.group(1)) if ep else None,
                }
                # Try to infer date from file content or fallback to directory name/mtime
                date_match = re.search(r"Training Date\s*:\s*([0-9_]+)", text)
                date_key = date_match.group(1) if date_match else ""
                if any(v is not None for v in params.values()):
                    return params, date_key
                return None, date_key
            except Exception:
                return None, ""

        # Build enriched list with both README and metadata
        enriched: list[tuple[Path, dict | None, str, dict | None, str, float]] = []
        for d in candidates:
            rd_params, rd_date = parse_readme(d)
            md_params, md_date = read_metadata(d)
            mtime = d.stat().st_mtime
            enriched.append((d, rd_params, rd_date, md_params, md_date, mtime))

        # Sort by date preference (README date first), fall back to metadata date, then mtime
        def sort_key(item):
            d, rd_params, rd_date, md_params, md_date, mtime = item
            if rd_date:
                return (2, rd_date)
            if md_date:
                return (1, md_date)
            return (0, f"{mtime:020.3f}")

        enriched.sort(key=sort_key, reverse=True)

        for d, rd_params, rd_date, md_params, md_date, mtime in enriched:
            # Prefer README params if complete
            if rd_params and all(v is not None for v in rd_params.values()):
                rd_params["source_dir"] = str(d)
                rd_params["source"] = "README.md"
                return rd_params
            # Else try metadata if complete
            if md_params and all(v is not None for v in md_params.values()):
                md_params["source_dir"] = str(d)
                md_params["source"] = "metadata.json"
                return md_params
            # Else combine if partial information available
            combined = {}
            if rd_params:
                combined.update({k: v for k, v in rd_params.items() if v is not None})
            if md_params:
                for k, v in md_params.items():
                    combined.setdefault(k, v)
            if combined and all(k in combined and combined[k] is not None for k in ("batch_size", "learning_rate", "epochs")):
                combined["source_dir"] = str(d)
                combined["source"] = "README+metadata"
                return combined

        return None

    @staticmethod
    def apply_hyperparams_to_config(config, params: dict):
        """Apply loaded hyperparameters into the config object in-place."""
        if not params:
            return
        if params.get("batch_size"):
            config.batch_size = int(params["batch_size"])
        if params.get("learning_rate"):
            config.learning_rate = float(params["learning_rate"])
        if params.get("epochs"):
            config.epochs = int(params["epochs"])

# Standalone utility functions
def get_device_info():
    """Get device information"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_info = {
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

    if torch.cuda.is_available():
        device_info['device_name'] = torch.cuda.get_device_name(0)
        device_info['cuda_version'] = torch.version.cuda

    return device_info

def ensure_reproducibility(seed=2002):
    """Ensure reproducibility across runs"""
    Utils.set_seed(seed)
    # Additional PyTorch settings for reproducibility
    torch.use_deterministic_algorithms(True, warn_only=True)


class Visualizer:
    """Class for creating visualizations"""

    @staticmethod
    def plot_model_comparison(results_dict, metric='accuracy', save_path=None):
        """
        Plot comparison of models based on a specific metric

        Args:
            results_dict: Dictionary with model names as keys and metrics as values
            metric: Metric to compare (default: 'accuracy')
            save_path: Path to save the plot
        """
        models = list(results_dict.keys())
        values = [results_dict[model].get(metric, 0) for model in models]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, values, alpha=0.7, color='steelblue')
        plt.title(f'Model Comparison - {metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (bar, v) in enumerate(zip(bars, values)):
            plt.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Model comparison plot saved to: {save_path}")

        plt.close()

    @staticmethod
    def plot_training_history(history, save_path=None):
        """Plot training history (loss and accuracy)"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot loss
        ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
        ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Plot accuracy
        ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        ax2.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Training history plot saved to: {save_path}")

        plt.close()


class Logger:
    """Simple logging class"""

    def __init__(self, log_file=None):
        self.log_file = log_file

    def log(self, message):
        """Log a message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)

        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(log_message + '\n')

    def log_dict(self, data_dict, prefix=""):
        """Log a dictionary"""
        if prefix:
            self.log(prefix)
        for key, value in data_dict.items():
            self.log(f"  {key}: {value}")


class DeviceManager:
    """Manage device (CPU/GPU) selection and information"""

    @staticmethod
    def get_device():
        """Get the best available device (CUDA if available, else CPU)"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"[OK] Using device: {device} ({torch.cuda.get_device_name(0)})")
        else:
            device = torch.device('cpu')
            print(f"[OK] Using device: {device}")
        return device

    @staticmethod
    def get_device_info():
        """Get detailed device information"""
        info = {
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        }

        if torch.cuda.is_available():
            info['device_name'] = torch.cuda.get_device_name(0)
            info['cuda_version'] = torch.version.cuda
            info['memory_allocated'] = torch.cuda.memory_allocated(0)
            info['memory_reserved'] = torch.cuda.memory_reserved(0)

        return info

    @staticmethod
    def print_device_info():
        """Print device information"""
        info = DeviceManager.get_device_info()
        print("=" * 50)
        print("DEVICE INFORMATION")
        print("=" * 50)
        print(f"CUDA Available: {info['cuda_available']}")

        if info['cuda_available']:
            print(f"Device Count: {info['device_count']}")
            print(f"Current Device: {info['current_device']}")
            print(f"Device Name: {info['device_name']}")
            print(f"CUDA Version: {info['cuda_version']}")
            print(f"Memory Allocated: {info['memory_allocated'] / 1024**2:.2f} MB")
            print(f"Memory Reserved: {info['memory_reserved'] / 1024**2:.2f} MB")
        else:
            print("Running on CPU")
        print("=" * 50)

    @staticmethod
    def clear_cache():
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[OK] GPU cache cleared")

    @staticmethod
    def fix_miopen_cache(clear_cache=True, disable_cache=False):
        """
        Fix MIOpen SQLite database error by clearing corrupted cache
        This must be called BEFORE any GPU operations

        Args:
            clear_cache: Whether to clear existing cache directories
            disable_cache: Whether to disable MIOpen cache (reduces performance but fixes compatibility issues)
                          False = PERFORMANCE MODE (cache enabled, faster)
                          True = COMPATIBILITY MODE (cache disabled, slower but more stable)
        """
        import shutil
        import tempfile

        print("=" * 80)
        print(" FIXING MIOPEN CACHE ERROR")
        print("=" * 80)

        if clear_cache:
            # Possible MIOpen cache locations
            cache_paths = []

            # Windows paths
            if os.name == 'nt':
                username = os.getenv("USERNAME", "")
                cache_paths.extend([
                    Path.home() / ".cache" / "miopen",
                    Path(f"C:/Users/{username}/.cache/miopen") if username else None,
                    Path(os.getenv("LOCALAPPDATA", "")) / "AMD" / "MIOpen" if os.getenv("LOCALAPPDATA") else None,
                    Path(os.getenv("APPDATA", "")) / "AMD" / "MIOpen" if os.getenv("APPDATA") else None,
                    Path(os.getenv("TEMP", tempfile.gettempdir())) / "miopen",
                ])
            # Linux paths
            else:
                cache_paths.extend([
                    Path.home() / ".config" / "miopen",
                    Path.home() / ".cache" / "miopen",
                    Path("/tmp/miopen-cache"),
                    Path("/var/tmp/miopen-cache"),
                ])

            # Filter out None values
            cache_paths = [p for p in cache_paths if p is not None]

            cleared = False
            for cache_path in cache_paths:
                if cache_path.exists():
                    try:
                        print(f"    Found MIOpen cache: {cache_path}")
                        shutil.rmtree(cache_path)
                        print(f"   [OK] Deleted: {cache_path}")
                        cleared = True
                    except Exception as e:
                        print(f"   [WARNING]  Could not delete {cache_path}: {e}")

            if cleared:
                print("   [OK] MIOpen cache cleared successfully!")
            else:
                print("   [INFO]  No MIOpen cache found (may already be clean)")

        # Configure environment variables based on disable_cache flag
        if disable_cache:
            # COMPATIBILITY MODE: Disable cache (slower but more compatible)
            os.environ["MIOPEN_DISABLE_CACHE"] = "1"
            os.environ["MIOPEN_FIND_MODE"] = "NORMAL"
            os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"
            os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = ""
            print("     MIOpen cache DISABLED (compatibility mode - slower)")
        else:
            # PERFORMANCE MODE: Enable cache (faster)
            # Only set FIND_MODE to ensure consistent behavior
            os.environ["MIOPEN_FIND_MODE"] = "NORMAL"
            # Explicitly remove cache-disabling variables if they were set
            os.environ.pop("MIOPEN_DISABLE_CACHE", None)
            os.environ.pop("MIOPEN_DEBUG_DISABLE_FIND_DB", None)
            os.environ.pop("MIOPEN_CUSTOM_CACHE_DIR", None)
            print("    MIOpen cache ENABLED (performance mode - faster)")

        print("   [OK] MIOpen environment variables configured")
        print("=" * 80)

        return clear_cache if clear_cache else False

    @staticmethod
    def run_additional_miopen_fix(script_path):
        """Run an external MIOpen fix script if it exists"""
        script_path = Path(script_path)
        if not script_path.exists():
            print(f"[INFO]  Skipping optional MIOpen fix – script not found: {script_path}")
            return False

        print(f"  Running optional MIOpen fix script: {script_path.name}")
        try:
            result = subprocess.run([sys.executable, str(script_path)], check=True)
            print(f"[OK] Script completed with return code {result.returncode}")
            return True
        except subprocess.CalledProcessError as exc:
            print(f"[WARNING]  Optional script failed ({script_path.name}): {exc}")
            return False

    @staticmethod
    def apply_miopen_fixes(config):
        """Apply configured MIOpen fixes based on current settings"""
        if not config.enable_miopen_fix:
            print("[INFO]  MIOpen fix disabled via configuration")
            return

        base_dir = Path(__file__).parent.parent
        sources = config.miopen_fix_sources

        if sources.get('device_manager', True):
            # Pass the disable_cache flag from config for performance optimization
            # Only clear cache when disabling it (compatibility mode)
            # Keep cache when enabling it (performance mode)
            DeviceManager.fix_miopen_cache(
                clear_cache=config.miopen_disable_cache,
                disable_cache=config.miopen_disable_cache
            )

        if sources.get('fix_miopen_amd'):
            DeviceManager.run_additional_miopen_fix(base_dir / "fix_miopen_amd.py")

        if sources.get('fix_miopen_error'):
            DeviceManager.run_additional_miopen_fix(base_dir / "fix_miopen_error.py")


# Test function for independent execution
def test_utils_module():
    """Test the utils module independently"""
    print("Testing Utils module...")

    try:
        # Test system info
        print("Testing system info...")
        Utils.print_system_info()
        print("[OK] System info printed successfully")

        # Test device info
        print("Testing device info...")
        device_info = get_device_info()
        print(f"Device info: {device_info}")
        print("[OK] Device info retrieved successfully")

        # Test seed setting
        print("Testing seed setting...")
        Utils.set_seed(2002)
        print("[OK] Seed set successfully")

        # Test timestamp
        print("Testing timestamp...")
        timestamp = Utils.get_timestamp()
        print(f"Current timestamp: {timestamp}")
        print("[OK] Timestamp generated successfully")

        # Test time formatting
        print("Testing time formatting...")
        test_times = [45, 125, 3665]
        for t in test_times:
            formatted = Utils.format_time(t)
            print(f"{t} seconds = {formatted}")
        print("[OK] Time formatting successful")

        # Test directory creation (use temp directory)
        print("Testing directory creation...")
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_utils"
            Utils.create_directory(test_dir)
            assert test_dir.exists(), "Directory was not created"
            print("[OK] Directory creation successful")

            # Test JSON save/load
            print("Testing JSON operations...")
            test_data = {"test": "data", "number": 123}
            json_file = test_dir / "test.json"
            Utils.save_json(test_data, json_file)
            loaded_data = Utils.load_json(json_file)
            assert loaded_data == test_data, "JSON data mismatch"
            print("[OK] JSON operations successful")

            # Test experiment directory creation
            print("Testing experiment directory creation...")
            exp_dir = Utils.create_experiment_dir(test_dir, "test_experiment")
            assert exp_dir.exists(), "Experiment directory not created"
            assert (exp_dir / "models").exists(), "Models subdirectory not created"
            assert (exp_dir / "plots").exists(), "Plots subdirectory not created"
            assert (exp_dir / "logs").exists(), "Logs subdirectory not created"
            assert (exp_dir / "results").exists(), "Results subdirectory not created"
            print("[OK] Experiment directory creation successful")

        # Test parameter counting (with dummy model)
        print("Testing parameter counting...")
        dummy_model = torch.nn.Linear(10, 5)
        param_counts = Utils.count_parameters(dummy_model)
        expected_total = 10 * 5 + 5  # weights + biases
        assert param_counts['total_parameters'] == expected_total, f"Parameter count mismatch: expected {expected_total}, got {param_counts['total_parameters']}"
        print("[OK] Parameter counting successful")

        # Test reproducibility
        print("Testing reproducibility...")
        ensure_reproducibility(2002)
        print("[OK] Reproducibility settings applied")

        print("[OK] All utils tests successful")
        print("[OK] Utils module test PASSED")
        return True

    except Exception as e:
        print(f"[FAIL] Utils module test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_utils_module()
    print(f"\nUtils Module Test Result: {'PASS' if success else 'FAIL'}")
