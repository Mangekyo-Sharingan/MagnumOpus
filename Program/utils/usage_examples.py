"""
EXAMPLE: How to Use the Excel Logger in Your Programs

This file demonstrates how to integrate the ExcelLogger into your existing
MagnumOpus project programs.
"""

import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Example 1: Basic usage in any program
def example_basic_usage():
    """Simple logging example"""
    from excel_logger import ExcelLogger

    with ExcelLogger("my_program") as logger:
        logger.log("Program started")
        logger.log("Processing data...")
        logger.log("Program completed")


# Example 2: Using in the training module
def example_training_integration():
    """Example of how to integrate with training.py"""
    from excel_logger import ExcelLogger

    with ExcelLogger("training") as logger:
        logger.log("Starting training process")
        logger.log_separator()

        # Log configuration
        logger.log("Configuration:")
        logger.log_dict({
            "model": "ResNet50",
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50
        })
        logger.log_separator()

        # Log each epoch
        for epoch in range(1, 4):  # Example with 3 epochs
            logger.log(f"Epoch {epoch}/50")
            logger.log_metrics({
                "train_loss": 0.5 - epoch*0.1,
                "train_acc": 0.7 + epoch*0.05,
                "val_loss": 0.6 - epoch*0.08,
                "val_acc": 0.65 + epoch*0.05
            })

        logger.log_separator()
        logger.log("Training completed successfully!")


# Example 3: Using in the data loading module
def example_data_loading():
    """Example of how to integrate with data.py"""
    from excel_logger import ExcelLogger

    with ExcelLogger("data_loading") as logger:
        logger.log("Starting data loading process")

        logger.log("Loading APTOS dataset...")
        logger.log_dict({
            "train_samples": 3662,
            "test_samples": 1928,
            "classes": 5
        }, prefix="APTOS Dataset Info:")

        logger.log("Loading EyePACS dataset...")
        logger.log_dict({
            "train_samples": 35126,
            "classes": 5
        }, prefix="EyePACS Dataset Info:")

        logger.log_separator()
        logger.log("Merging datasets...")
        logger.log("Total samples: 38,788")
        logger.log("Data loading completed!")


# Example 4: Using in the evaluation module
def example_evaluation():
    """Example of how to integrate with test.py"""
    from excel_logger import ExcelLogger

    with ExcelLogger("evaluation") as logger:
        logger.log("Starting model evaluation")
        logger.log("Model: ResNet50")
        logger.log_separator()

        logger.log_metrics({
            "overall_accuracy": 0.8745,
            "precision": 0.8612,
            "recall": 0.8534,
            "f1_score": 0.8573
        })

        logger.log_separator("-", 60)
        logger.log("Per-class Performance:")

        classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
        for i, class_name in enumerate(classes):
            logger.log(f"Class {i} ({class_name}):")
            logger.log_dict({
                "precision": 0.85 + i*0.02,
                "recall": 0.83 + i*0.015,
                "f1-score": 0.84 + i*0.018,
                "support": 500 + i*100
            })


# Example 5: Using ExcelLoggerMulti for testing multiple modules
def example_multi_module_testing():
    """Example of logging multiple programs in one session"""
    from excel_logger import ExcelLoggerMulti

    # Create multi-logger
    multi_logger = ExcelLoggerMulti(excel_file="module_tests.xlsx")

    # Test config module
    config_log = multi_logger.get_logger("config_test")
    config_log.log("Testing config module")
    config_log.log("✓ All config tests passed")

    # Test data module
    data_log = multi_logger.get_logger("data_test")
    data_log.log("Testing data module")
    data_log.log("✓ Data loading test passed")
    data_log.log("✓ Data preprocessing test passed")

    # Test models module
    models_log = multi_logger.get_logger("models_test")
    models_log.log("Testing models module")
    models_log.log_list([
        "✓ VGG16 test passed",
        "✓ ResNet50 test passed",
        "✓ InceptionV3 test passed"
    ], title="Model Tests:")

    # Save all logs
    multi_logger.save_all()
    print("All module tests logged!")


# Example 6: Integration with existing run_module_tests.py
def example_integration_with_module_tests():
    """
    Example showing how to modify run_module_tests.py to use the logger

    In your run_module_tests.py, you would add:
    """
    code_example = '''
    # Add to the top of run_module_tests.py:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / "utils"))
    from excel_logger import ExcelLogger
    
    # In the run_all_tests() method:
    with ExcelLogger("module_tests") as logger:
        logger.log("Starting module tests")
        
        for module_name in self.modules_to_test:
            logger.log(f"Testing {module_name} module...")
            
            try:
                # Run test
                success = self.test_module(module_name)
                
                if success:
                    logger.log(f"✓ {module_name}: PASSED")
                else:
                    logger.log(f"✗ {module_name}: FAILED")
                    
            except Exception as e:
                logger.log(f"✗ {module_name}: ERROR - {str(e)}")
        
        logger.log_separator()
        logger.log("Module testing completed")
    '''
    print("See the code example above for integration pattern")


# Example 7: Capturing all stdout (optional advanced usage)
def example_stdout_capture():
    """Example of capturing all print statements automatically"""
    from excel_logger import ExcelLogger

    # Set capture_stdout=True to automatically capture all print statements
    with ExcelLogger("stdout_capture_test", capture_stdout=True) as logger:
        # These print statements will be captured
        print("This will be captured in the log")
        print("Performing calculations...")
        result = 42 * 2
        print(f"Result: {result}")

        # You can still use logger.log() for structured logging
        logger.log_dict({"result": result, "status": "success"})


if __name__ == "__main__":
    print("="*80)
    print("EXCEL LOGGER - USAGE EXAMPLES")
    print("="*80)

    print("\nRunning example 1: Basic usage...")
    example_basic_usage()

    print("\nRunning example 2: Training integration...")
    example_training_integration()

    print("\nRunning example 3: Data loading...")
    example_data_loading()

    print("\nRunning example 4: Evaluation...")
    example_evaluation()

    print("\nRunning example 5: Multi-module testing...")
    example_multi_module_testing()

    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED!")
    print("="*80)
    print("\nCheck your logs directory:")
    print("  C:\\Users\\adamh\\PycharmProjects\\MagnumOpus\\Program\\logs\\")
    print("\nYou should see:")
    print("  - program_logs.xlsx (with worksheets for each example)")
    print("  - module_tests.xlsx (from multi-module example)")
