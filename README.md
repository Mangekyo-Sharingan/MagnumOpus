# Diabetic Retinopathy Classification - MagnumOpus Project

## Table of Contents
- [Latest Features](#latest-features)
- [Project Overview](#project-overview)
- [Project Structure](#project-structure---may-vary)
- [Quick Start](#quick-start)
- [Testing the Project](#testing-the-project)
- [Module Documentation](#module-documentation)
- [Supported Models](#supported-models)
- [Classification Classes](#classification-classes)
- [Training Process](#training-process)
- [Testing Philosophy](#testing-philosophy)
- [Development Workflow](#development-workflow)
- [Recent Changes](#recent-changes)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [AI Usage](#ai-usage)
- [License](#license)

## Latest Features

### Automatic Hyperparameter Tuning (Nov 11, 2025)
- **Grid search before training** - Automatically find best batch size & learning rate
- **Model-specific tuning** - Each architecture gets optimal hyperparameters
- **Three modes**: SMART (fast), Random, or Full Grid search
- **Fixed random state** - Fair comparison across all combinations
- **Analysis across differently sized subsets of data**

### Interactive Model Selection (Nov 11, 2025)
- **Select specific models** to train before starting
- **Train all or choose individual models** (VGG16, ResNet50, InceptionV3)
- **Flexible input options**: numbers, comma-separated, or 'all'

### Key features
- **Hyperparameter Tuning** - Auto-optimize batch size & learning rate (NEW!)
- **Interactive Model Selection** - Choose which models to train
- **Complete Model Saving** - Save trained models for future predictions
- **Progress Tracking** - Real-time training progress with time estimates
- **Pause Between Models** - Review results before continuing
- **Model Loading** - Easy loading of trained models for predictions
- **Comprehensive Logging** - Detailed metrics and history

---

## Project Overview

This project implements deep learning models to classify diabetic retinopathy severity using retinal fundus images. The system supports multiple CNN architectures and provides a complete pipeline from data preprocessing to model evaluation.

### Key Features

- **Multiple CNN Architectures**: VGG16, ResNet50, and InceptionV3
- **Dual Dataset Support**: APTOS 2019 and EyePACS datasets
- **Advanced Data Pipeline**: Lazy loading, data augmentation, and preprocessing
- **Comprehensive Training**: Full training loop with validation and checkpointing
- **Detailed Evaluation**: Metrics, confusion matrices, and misclassification analysis
- **Independent Module Testing**: Each component can be tested individually
- **PyTorch Implementation**: Modern, efficient deep learning framework

## Project Structure - may vary

```
MagnumOpus/
├── README.md                   # This file
├── Data/                       # Dataset storage
│   ├── Aptos/                 # APTOS 2019 dataset
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── train_images/
│   │   └── test_images/
│   └── EyePacs/               # EyePACS dataset
│       ├── trainLabels.csv
│       ├── train/
│       └── test/
├── Program/                    # Main program directory
│   ├── main.py                # Main entry point
│   ├── run_module_tests.py    # Master test script
│   ├── modules/               # Core modules
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration management
│   │   ├── data.py            # Data loading and preprocessing
│   │   ├── models.py          # CNN model implementations
│   │   ├── train.py           # Training pipeline
│   │   ├── test.py            # Evaluation and testing
│   │   └── utils.py           # Utility functions
│   └── tests/                 # Test results and logs
```

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install torch torchvision
pip install pandas numpy scikit-learn
pip install matplotlib seaborn
pip install pillow
pip install ray[train,tune]  # For distributed execution
```

### Running the Complete Pipeline

```bash
cd Program
python main.py
```

This will:
1. Set up project directories
2. Load and prepare data
3. Show model selection menu
4. Initialize selected models
5. Train each model with progress tracking
6. Save complete model packages
7. Evaluate and compare results

### Example: Train Specific Models

When prompted during `python main.py`:

**Train all models:**
```
> Select models to train: all
```

**Train only VGG16:**
```
> Select models to train: 1
```

**Train VGG16 and InceptionV3:**
```
> Select models to train: 1,3
```

### Load a Trained Model for Predictions

```python
import torch
from modules import ModelFactory, Config

# Setup
config = Config()
model = ModelFactory.create_model('vgg16', config)

# Load trained weights
checkpoint = torch.load('../models/vgg16_20251111_103727/vgg16_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Ready for predictions!
```

See [USAGE_GUIDE.md](USAGE_GUIDE.md) for complete examples.

---

## Batch Mode & Distributed Execution

The project supports non-interactive batch execution for use with Ray clusters, remote job submission, and CI/CD pipelines.

### Command Line Interface

```bash
cd Program
python main.py --help
```

**Available arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--models` | Models to train (vgg16, resnet50, inceptionv3, or all) | all |
| `--epochs` | Number of training epochs | 50 |
| `--batch-size` | Training batch size | 32 |
| `--learning-rate` | Learning rate | 0.0001 |
| `--num-workers` | DataLoader workers | 4 |
| `--batch-mode` | Force batch mode (non-interactive) | False |
| `--data-dir` | Data directory path | ../Data |
| `--output-dir` | Model output directory | ../models |
| `--mode` | Execution mode: train, tune, evaluate | train |

### Examples

**Train all models (interactive):**
```bash
python main.py
```

**Train specific models in batch mode:**
```bash
python main.py --models vgg16 resnet50 --epochs 100 --batch-mode
```

**Hyperparameter tuning:**
```bash
python main.py --mode tune --models inceptionv3 --batch-mode
```

### Environment Variable Configuration

All training parameters can be configured via environment variables for Ray job submission:

| Environment Variable | Description |
|---------------------|-------------|
| `RAY_EPOCHS` | Number of training epochs |
| `RAY_BATCH_SIZE` | Training batch size |
| `RAY_LEARNING_RATE` | Learning rate |
| `RAY_NUM_WORKERS` | DataLoader worker count |
| `RAY_MODELS` | Comma-separated list of models |
| `RAY_DATA_DIR` | Data directory path |
| `RAY_OUTPUT_DIR` | Output directory path |

**Example Ray job submission:**
```bash
RAY_EPOCHS=100 RAY_BATCH_SIZE=64 RAY_MODELS=vgg16,resnet50 python main.py --batch-mode
```

### Ray Cluster Execution

The code automatically detects when running inside a Ray worker and enables batch mode:

```python
import ray

@ray.remote
def train_model():
    # main.py will auto-detect Ray environment
    import subprocess
    subprocess.run(["python", "main.py", "--models", "vgg16", "--batch-mode"])
```

### Batch Mode Behavior

When running in batch mode:
- **No interactive prompts** - All selections made via CLI/env vars
- **Reduced console output** - Only essential progress updates
- **No live plotting** - Training plots saved to disk only
- **Auto-selection** - Uses specified models without confirmation
- **Optimized DataLoader** - Prefetching and persistent workers enabled

---


## Testing the Project

### Test All Modules at Once
```bash
cd Program
python run_module_tests.py
```

### Test Individual Modules
```bash
cd Program/modules
python config.py      # Test configuration
python data.py        # Test data pipeline
python models.py      # Test model architectures
python train.py       # Test training components
python test.py        # Test evaluation system
python utils.py       # Test utility functions
```

### Expected Output

When you run the tests, you'll see detailed feedback like:
```
================================================================================
DIABETIC RETINOPATHY PROJECT - MODULE TESTING
================================================================================
Test started at: 2025-10-13 15:30:45
Testing 6 modules...
================================================================================

==================== TESTING CONFIG MODULE ====================
[OK] Successfully imported config module
[OK] Found test function: test_config_module
Testing Config module...
[OK] Config object created successfully
[OK] Base directory exists: True
[OK] Data directory exists: True
...
[OK] Config module test PASSED
[OK] CONFIG MODULE TEST: PASSED

...

ALL MODULES PASSED!
The project structure is correctly implemented with PyTorch.
All modules can be run independently and work as expected.
```

##  Module Documentation

### Configuration Module (`config.py`)
Manages all project settings including:
- Dataset paths and configurations
- Model hyperparameters
- Training parameters
- Hardware settings

### Data Module (`data.py`)
Comprehensive data pipeline featuring:
- **CustomDataLoader**: Lazy loading for memory efficiency
- **Pipeline A & B**: Different preprocessing for different models
- **Dataset Merging**: Combines APTOS and EyePACS datasets
- **Comprehensive Logging**: Detailed progress tracking

### Models Module (`models.py`)
CNN implementations including:
- **VGG16Model**: Transfer learning with custom classifier
- **ResNet50Model**: Deep residual network implementation
- **InceptionV3Model**: Advanced architecture with auxiliary outputs
- **BaseModel**: Abstract base class for consistency

### Training Module (`train.py`)
Complete training pipeline with:
- **Trainer Class**: Handles entire training process
- **Advanced Optimization**: Adam optimizer with learning rate scheduling
- **Checkpointing**: Save and resume training
- **Validation**: Built-in validation during training

### Testing Module (`test.py`)
Comprehensive evaluation system:
- **Evaluator Class**: Complete model evaluation
- **Metrics Calculation**: Accuracy, precision, recall, F1-score
- **Visualization**: Confusion matrices and distribution plots
- **Misclassification Analysis**: Identify problematic samples

### Utilities Module (`utils.py`)
Helper functions including:
- **System Information**: Hardware and software details
- **Reproducibility**: Seed setting for consistent results
- **File Operations**: JSON/pickle save/load
- **Experiment Management**: Directory structure creation

## Supported Models

### VGG16
- **Input Size**: 224×224
- **Use Case**: Baseline model, good for initial experiments
- **Features**: Pre-trained on ImageNet, custom classifier

### ResNet50
- **Input Size**: 224×224
- **Use Case**: Better performance than VGG16, residual connections
- **Features**: Skip connections, deeper architecture

### InceptionV3
- **Input Size**: 299×299
- **Use Case**: Best performance, multi-scale feature extraction
- **Features**: Auxiliary outputs, inception modules

## Classification Classes

The models classify diabetic retinopathy into 5 severity levels:
- **Class 0**: No DR (No Diabetic Retinopathy)
- **Class 1**: Mild DR
- **Class 2**: Moderate DR
- **Class 3**: Severe DR
- **Class 4**: Proliferative DR

## Training Process

1. **Data Loading**: Merge APTOS and EyePACS datasets
2. **Preprocessing**: Apply appropriate pipeline (A or B)
3. **Model Creation**: Initialize chosen architecture
4. **Training Loop**: Train with validation monitoring
5. **Checkpointing**: Save best model based on validation loss
6. **Evaluation**: Comprehensive testing on hold-out set

## Testing Philosophy

Each module includes comprehensive testing:
- **Independence**: Each module can run standalone
- **Comprehensive Coverage**: Tests all major functionality
- **Clear Feedback**: Detailed success/failure messages
- **Error Handling**: Graceful failure with informative messages

## Development Workflow

### Adding New Features
1. Implement functionality in appropriate module
2. Add corresponding test function
3. Run individual module test: `python module_name.py`
4. Run full test suite: `python run_module_tests.py`

### Debugging Issues
1. Run individual module tests to isolate problems
2. Check detailed error messages in test output
3. Use logging output from data pipeline for data issues
4. Examine model summaries for architecture problems

## Recent Changes

### PyTorch Migration (October 2025)
- **Complete Refactoring**: Migrated from TensorFlow to PyTorch
- **Enhanced Testing**: Added comprehensive module testing
- **Improved Logging**: Detailed progress tracking throughout pipeline
- **Better Architecture**: Cleaner, more maintainable code structure

### Key Improvements
- **Memory Efficiency**: Lazy loading for large datasets
- **Reproducibility**: Proper seed management
- **Modularity**: Each component is independently testable
- **Comprehensive Feedback**: No more silent failures

## Usage Examples

### Basic Training
```python
from modules.config import Config
from modules.data import CustomDataLoader
from modules.models import get_model
from modules.train import Trainer

# Setup
config = Config()
data_loader = CustomDataLoader(config)
data_loader.load_data()

# Create model and trainer
model = get_model('resnet50', config)
trainer = Trainer(model, config)

# Get data loaders
train_loader, val_loader = data_loader.create_data_loaders('resnet50')

# Train
trainer.train(train_loader, val_loader)
```

### Model Evaluation
```python
from modules.test import Evaluator

# Create evaluator
evaluator = Evaluator(model, config)

# Evaluate on test set
test_loader = data_loader.create_test_loader('resnet50')
predictions, true_labels, probs = evaluator.evaluate_model(test_loader)

# Generate comprehensive report
metrics, misclassified = evaluator.generate_report()
```

## Troubleshooting

### Common Issues

**"Process finished with exit code 0" (No Output)**
- **Solution**: This was the old behavior. The new modules provide comprehensive output.
- **Test**: Run `python run_module_tests.py` to see detailed progress.

**Module Import Errors**
- **Check**: Ensure you're in the correct directory
- **Run**: Individual module tests to isolate the issue

**CUDA/GPU Issues**
- **Check**: Run `python modules/utils.py` to see system information
- **Note**: Code automatically falls back to CPU if CUDA unavailable

**Data Loading Issues**
- **Check**: Verify dataset paths in config.py
- **Run**: `python modules/data.py` for detailed data loading feedback

## References

- [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection)
- [EyePACS Dataset](https://www.kaggle.com/c/diabetic-retinopathy-detection)
- [PyTorch Documentation](https://pytorch.org/docs/)

## AI Usage

Artificial Intelligence tools were used in the development of this project. Specifically:
- **Code Comments**: AI was utilized to generate explanatory comments throughout the codebase to enhance readability and maintainability.
- **README Generation**: Portions of the README files and documentation were generated or refined using AI assistance to ensure clarity and completeness.

## License

This project is for educational and research purposes. Please respect the original dataset licenses.
