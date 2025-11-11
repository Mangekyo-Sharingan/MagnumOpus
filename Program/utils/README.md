# Utils Module - Excel Logger

## Overview

The **utils** subfolder contains utility modules for the MagnumOpus project. The primary module is **ExcelLogger**, which provides comprehensive logging functionality that saves all program outputs to Excel files.

## Key Features

✅ **Separate Worksheets**: Each program/module gets its own worksheet  
✅ **Timestamped Runs**: Every run is timestamped and appended (no data loss!)  
✅ **No Overwrites**: Old logs are preserved; new runs are appended below  
✅ **Multiple Data Types**: Log strings, dictionaries, lists, metrics, etc.  
✅ **Automatic Formatting**: Headers, colors, and column widths auto-adjusted  
✅ **Easy Integration**: Simple context manager interface  
✅ **Fallback Safety**: Falls back to text files if Excel fails  

## Installation

The module automatically installs `openpyxl` if it's not already installed. No manual setup required!

## Quick Start

### Basic Usage

```python
from utils.excel_logger import ExcelLogger

with ExcelLogger("my_program") as logger:
    logger.log("Program started")
    logger.log("Processing data...")
    logger.log("Program completed")
```

This creates a worksheet named "my_program" in `Program/logs/program_logs.xlsx` and logs all messages with timestamps.

## API Reference

### ExcelLogger Class

#### Constructor

```python
ExcelLogger(program_name, excel_file=None, log_dir=None, capture_stdout=False)
```

**Parameters:**
- `program_name` (str): Name of the program/module being logged
- `excel_file` (str, optional): Excel filename (default: "program_logs.xlsx")
- `log_dir` (str, optional): Log directory (default: "Program/logs/")
- `capture_stdout` (bool, optional): Auto-capture print statements (default: False)

#### Methods

##### `log(message, data=None)`
Log a text message with optional structured data.

```python
logger.log("Training started")
logger.log("Epoch completed", {"epoch": 1, "loss": 0.5})
```

##### `log_dict(data_dict, prefix="")`
Log a dictionary in readable format.

```python
logger.log_dict({
    "accuracy": 0.85,
    "precision": 0.83,
    "recall": 0.87
}, prefix="Model Performance:")
```

##### `log_list(data_list, title="List Data")`
Log a list of items.

```python
logger.log_list([
    "Test 1: PASSED",
    "Test 2: PASSED",
    "Test 3: FAILED"
], title="Test Results")
```

##### `log_metrics(metrics_dict)`
Log metrics in structured format (ideal for training/evaluation).

```python
logger.log_metrics({
    "train_loss": 0.45,
    "train_acc": 0.85,
    "val_loss": 0.52,
    "val_acc": 0.82
})
```

##### `log_separator(char="=", length=80)`
Log a separator line.

```python
logger.log_separator()      # Default: ========...
logger.log_separator("-", 60)  # Custom: ------... (60 chars)
```

### ExcelLoggerMulti Class

For logging multiple programs in a single session.

```python
from utils.excel_logger import ExcelLoggerMulti

multi_logger = ExcelLoggerMulti(excel_file="multi_tests.xlsx")

config_logger = multi_logger.get_logger("config")
config_logger.log("Config test passed")

data_logger = multi_logger.get_logger("data")
data_logger.log("Data test passed")

multi_logger.save_all()  # Save all logs
```

## Usage Examples

### Example 1: Training Loop

```python
from utils.excel_logger import ExcelLogger

with ExcelLogger("training") as logger:
    logger.log("Starting training")
    logger.log_separator()
    
    logger.log("Configuration:")
    logger.log_dict({
        "model": "ResNet50",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50
    })
    
    for epoch in range(1, 51):
        # Training code here...
        
        logger.log(f"Epoch {epoch}/50")
        logger.log_metrics({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
    
    logger.log_separator()
    logger.log("Training completed!")
```

### Example 2: Data Loading

```python
from utils.excel_logger import ExcelLogger

with ExcelLogger("data_pipeline") as logger:
    logger.log("Loading APTOS dataset...")
    logger.log_dict({
        "train_samples": 3662,
        "test_samples": 1928,
        "classes": 5
    }, prefix="APTOS Stats:")
    
    logger.log("Loading EyePACS dataset...")
    logger.log_dict({
        "train_samples": 35126,
        "classes": 5
    }, prefix="EyePACS Stats:")
    
    logger.log("Merging datasets...")
    logger.log(f"Total samples: {total_samples}")
```

### Example 3: Model Evaluation

```python
from utils.excel_logger import ExcelLogger

with ExcelLogger("evaluation") as logger:
    logger.log(f"Evaluating {model_name}")
    logger.log_separator()
    
    logger.log_metrics({
        "overall_accuracy": 0.8745,
        "precision": 0.8612,
        "recall": 0.8534,
        "f1_score": 0.8573
    })
    
    logger.log_separator("-")
    logger.log("Per-class Performance:")
    
    for i, class_name in enumerate(class_names):
        logger.log(f"Class {i} ({class_name}):")
        logger.log_dict(per_class_metrics[i])
```

### Example 4: Module Testing

```python
from utils.excel_logger import ExcelLogger

with ExcelLogger("module_tests") as logger:
    logger.log("Starting module tests")
    logger.log_separator()
    
    for module_name in modules_to_test:
        logger.log(f"Testing {module_name}...")
        
        try:
            test_result = test_module(module_name)
            if test_result:
                logger.log(f"✓ {module_name}: PASSED")
            else:
                logger.log(f"✗ {module_name}: FAILED")
        except Exception as e:
            logger.log(f"✗ {module_name}: ERROR - {str(e)}")
    
    logger.log_separator()
    logger.log("Testing completed")
```

## Excel File Structure

The Excel files are organized as follows:

```
program_logs.xlsx
├── Sheet: "training"
│   ├── Run Date | Timestamp | Message | Data
│   ├── 2025-11-03 | 14:30:00 | === RUN: 2025-11-03 14:30:00 ===
│   ├── 2025-11-03 | 14:30:01 | Training started
│   ├── 2025-11-03 | 14:30:02 | Epoch 1/50
│   ├── 2025-11-03 | 14:30:02 | --- METRICS ---
│   ├── ... (more logs)
│   ├── 
│   ├── 2025-11-03 | 16:45:00 | === RUN: 2025-11-03 16:45:00 ===
│   ├── 2025-11-03 | 16:45:01 | Training started (2nd run)
│   └── ... (more logs)
│
├── Sheet: "evaluation"
│   └── ... (evaluation logs)
│
└── Sheet: "data_loading"
    └── ... (data loading logs)
```

**Key Points:**
- Each run is separated by a timestamped header
- Old data is NEVER overwritten
- Each program has its own worksheet
- Easy to compare runs by date/time

## Integration with Existing Code

### Integrating with `run_module_tests.py`

Add at the top of your file:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "utils"))
from excel_logger import ExcelLogger
```

Then wrap your test execution:

```python
def run_all_tests(self):
    with ExcelLogger("module_tests") as logger:
        logger.log("Starting module tests")
        
        for module_name in self.modules_to_test:
            logger.log(f"Testing {module_name}...")
            success = self.test_module(module_name)
            
            if success:
                logger.log(f"✓ {module_name}: PASSED")
            else:
                logger.log(f"✗ {module_name}: FAILED")
```

### Integrating with Training Module

In `modules/train.py`:

```python
# At the top
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from excel_logger import ExcelLogger

# In your Trainer class
def train(self, train_loader, val_loader):
    with ExcelLogger("training") as logger:
        logger.log(f"Training {self.model.model_name}")
        logger.log_dict(self.config.get_training_params())
        
        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            logger.log(f"Epoch {epoch+1}/{self.config.epochs}")
            logger.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })
```

## File Locations

- **Module Location**: `Program/utils/excel_logger.py`
- **Default Log Directory**: `Program/logs/`
- **Default Excel File**: `program_logs.xlsx`
- **Usage Examples**: `Program/utils/usage_examples.py`

## Testing

To test the Excel logger:

```bash
cd Program/utils
python excel_logger.py
```

To see usage examples:

```bash
cd Program/utils
python usage_examples.py
```

## Advanced Features

### Capturing All stdout (Optional)

If you want to automatically capture all print statements:

```python
with ExcelLogger("my_program", capture_stdout=True) as logger:
    print("This will be captured")
    print("And this too!")
    # All prints are automatically logged
```

### Custom Excel Files

Create separate Excel files for different purposes:

```python
# Training logs
with ExcelLogger("training", excel_file="training_logs.xlsx") as logger:
    logger.log("Training data here")

# Testing logs
with ExcelLogger("testing", excel_file="testing_logs.xlsx") as logger:
    logger.log("Testing data here")
```

### Custom Log Directories

```python
logger = ExcelLogger(
    "my_program",
    log_dir="C:/MyCustomLogs/"
)
```

## Troubleshooting

### Issue: "openpyxl not found"
**Solution**: The module auto-installs openpyxl, but you can manually install:
```bash
pip install openpyxl
```

### Issue: Excel file is locked
**Solution**: Close the Excel file before running your program

### Issue: Logs not appearing
**Solution**: Make sure to use the context manager (`with` statement) or the logs won't be saved

### Issue: Import errors
**Solution**: Make sure to add the utils directory to your path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "utils"))
from excel_logger import ExcelLogger
```

## Best Practices

1. **Use Context Manager**: Always use `with ExcelLogger(...) as logger:` to ensure logs are saved
2. **Descriptive Names**: Use clear program names for easy identification
3. **Log Regularly**: Log at key points in your program
4. **Use Separators**: Use `log_separator()` to make logs more readable
5. **Log Metrics**: Use `log_metrics()` for structured data
6. **Multiple Runs**: Don't worry about overwriting - runs are automatically appended

## Summary

The ExcelLogger utility provides a comprehensive solution for logging all program outputs in your MagnumOpus project:

- ✅ **No data loss** - all runs are preserved
- ✅ **Well organized** - separate worksheets per program
- ✅ **Timestamped** - easy to track when things happened
- ✅ **Easy to use** - simple API with context manager
- ✅ **Flexible** - works with any Python program
- ✅ **Formatted** - automatic styling and column widths

Start using it today to keep track of all your experiments, training runs, and test results!

---

**Created**: November 3, 2025  
**Version**: 1.0  
**Author**: MagnumOpus Utils Team

