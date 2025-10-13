"""
Utility functions for diabetic retinopathy classification project
"""
import os
import json
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

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
    def set_seed(seed=42):
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

        return total_params, trainable_params

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

def ensure_reproducibility(seed=42):
    """Ensure reproducibility across runs"""
    Utils.set_seed(seed)
    # Additional PyTorch settings for reproducibility
    torch.use_deterministic_algorithms(True, warn_only=True)

# Test function for independent execution
def test_utils_module():
    """Test the utils module independently"""
    print("Testing Utils module...")

    try:
        # Test system info
        print("Testing system info...")
        Utils.print_system_info()
        print("✓ System info printed successfully")

        # Test device info
        print("Testing device info...")
        device_info = get_device_info()
        print(f"Device info: {device_info}")
        print("✓ Device info retrieved successfully")

        # Test seed setting
        print("Testing seed setting...")
        Utils.set_seed(42)
        print("✓ Seed set successfully")

        # Test timestamp
        print("Testing timestamp...")
        timestamp = Utils.get_timestamp()
        print(f"Current timestamp: {timestamp}")
        print("✓ Timestamp generated successfully")

        # Test time formatting
        print("Testing time formatting...")
        test_times = [45, 125, 3665]
        for t in test_times:
            formatted = Utils.format_time(t)
            print(f"{t} seconds = {formatted}")
        print("✓ Time formatting successful")

        # Test directory creation (use temp directory)
        print("Testing directory creation...")
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_utils"
            Utils.create_directory(test_dir)
            assert test_dir.exists(), "Directory was not created"
            print("✓ Directory creation successful")

            # Test JSON save/load
            print("Testing JSON operations...")
            test_data = {"test": "data", "number": 123}
            json_file = test_dir / "test.json"
            Utils.save_json(test_data, json_file)
            loaded_data = Utils.load_json(json_file)
            assert loaded_data == test_data, "JSON data mismatch"
            print("✓ JSON operations successful")

            # Test experiment directory creation
            print("Testing experiment directory creation...")
            exp_dir = Utils.create_experiment_dir(test_dir, "test_experiment")
            assert exp_dir.exists(), "Experiment directory not created"
            assert (exp_dir / "models").exists(), "Models subdirectory not created"
            assert (exp_dir / "plots").exists(), "Plots subdirectory not created"
            assert (exp_dir / "logs").exists(), "Logs subdirectory not created"
            assert (exp_dir / "results").exists(), "Results subdirectory not created"
            print("✓ Experiment directory creation successful")

        # Test parameter counting (with dummy model)
        print("Testing parameter counting...")
        dummy_model = torch.nn.Linear(10, 5)
        total, trainable = Utils.count_parameters(dummy_model)
        expected_total = 10 * 5 + 5  # weights + biases
        assert total == expected_total, f"Parameter count mismatch: expected {expected_total}, got {total}"
        print("✓ Parameter counting successful")

        # Test reproducibility
        print("Testing reproducibility...")
        ensure_reproducibility(42)
        print("✓ Reproducibility settings applied")

        print("✓ All utils tests successful")
        print("✓ Utils module test PASSED")
        return True

    except Exception as e:
        print(f"✗ Utils module test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_utils_module()
    print(f"\nUtils Module Test Result: {'PASS' if success else 'FAIL'}")
