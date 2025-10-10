"""
Simplified test script for the data loading and preprocessing modules
This script validates that the data pipeline works correctly for training
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

# Add the modules directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import only the config module directly to avoid TensorFlow imports
from modules.config import Config

class SimpleDataTestSuite:
    """Simplified test suite for data loading functionality"""

    def __init__(self):
        self.config = Config()
        self.passed_tests = 0
        self.total_tests = 0

    def run_basic_tests(self):
        """Run basic data loading tests without TensorFlow dependencies"""
        print("="*60)
        print("BASIC DIABETIC RETINOPATHY DATA PIPELINE TESTS")
        print("="*60)

        # Test configuration
        self.test_config_setup()

        # Test basic data loading
        self.test_basic_data_loading()

        # Test image processing without generators
        self.test_basic_image_processing()

        # Summary
        self.print_test_summary()

    def test_config_setup(self):
        """Test that configuration is set up correctly"""
        print("\n--- Testing Configuration Setup ---")

        # Test 1: Check if paths exist
        self._test("APTOS data directory exists",
                  self.config.aptos_dir.exists())

        self._test("EyePACS data directory exists",
                  self.config.eyepacs_dir.exists())

        # Test 2: Check CSV files exist
        self._test("APTOS train.csv exists",
                  self.config.aptos_train_csv.exists())

        self._test("EyePACS trainLabels.csv exists",
                  self.config.eyepacs_train_csv.exists())

        # Test 3: Check model configurations
        expected_models = ['vgg16', 'resnet50', 'inceptionv3']
        self._test("All expected models configured",
                  all(model in self.config.model_configs for model in expected_models))

        # Test 4: Check image sizes
        self._test("VGG16 image size is 224x224",
                  self.config.get_image_size('vgg16') == (224, 224))

        self._test("InceptionV3 image size is 299x299",
                  self.config.get_image_size('inceptionv3') == (299, 299))

        # Test 5: Check pipeline assignments
        self._test("VGG16 uses Pipeline A",
                  self.config.get_pipeline_type('vgg16') == 'A')

        self._test("InceptionV3 uses Pipeline B",
                  self.config.get_pipeline_type('inceptionv3') == 'B')

    def test_basic_data_loading(self):
        """Test basic data loading without TensorFlow"""
        print("\n--- Testing Basic Data Loading ---")

        try:
            # Test APTOS CSV loading
            if self.config.aptos_train_csv.exists():
                aptos_df = pd.read_csv(self.config.aptos_train_csv)
                self._test(f"APTOS CSV loaded ({len(aptos_df)} samples)", len(aptos_df) > 0)

                # Check required columns
                required_cols = ['id_code', 'diagnosis']
                has_cols = all(col in aptos_df.columns for col in required_cols)
                self._test("APTOS CSV has required columns", has_cols)

                # Check diagnosis values are valid (0-4)
                valid_diagnoses = aptos_df['diagnosis'].isin([0, 1, 2, 3, 4]).all()
                self._test("APTOS diagnosis values are valid (0-4)", valid_diagnoses)

                print(f"    APTOS diagnosis distribution:")
                for diag, count in aptos_df['diagnosis'].value_counts().sort_index().items():
                    percentage = (count / len(aptos_df)) * 100
                    print(f"      Level {diag}: {count} ({percentage:.1f}%)")

            # Test EyePACS CSV loading
            if self.config.eyepacs_train_csv.exists():
                eyepacs_df = pd.read_csv(self.config.eyepacs_train_csv)
                self._test(f"EyePACS CSV loaded ({len(eyepacs_df)} samples)", len(eyepacs_df) > 0)

                # Check required columns
                required_cols = ['image', 'level']
                has_cols = all(col in eyepacs_df.columns for col in required_cols)
                self._test("EyePACS CSV has required columns", has_cols)

                # Check diagnosis values are valid (0-4)
                valid_levels = eyepacs_df['level'].isin([0, 1, 2, 3, 4]).all()
                self._test("EyePACS level values are valid (0-4)", valid_levels)

                print(f"    EyePACS level distribution:")
                for level, count in eyepacs_df['level'].value_counts().sort_index().items():
                    percentage = (count / len(eyepacs_df)) * 100
                    print(f"      Level {level}: {count} ({percentage:.1f}%)")

        except Exception as e:
            self._test(f"Data loading failed with error: {str(e)}", False)

    def test_basic_image_processing(self):
        """Test basic image processing functionality"""
        print("\n--- Testing Basic Image Processing ---")

        # Test if sample images exist and can be loaded
        sample_aptos_images = list(self.config.aptos_train_images_dir.glob("*.png"))[:5]
        sample_eyepacs_images = list(self.config.eyepacs_train_images_dir.glob("*.jpeg"))[:5]

        if sample_aptos_images:
            self._test(f"Found APTOS sample images ({len(sample_aptos_images)})", True)

            # Test loading a sample APTOS image
            try:
                sample_image = Image.open(sample_aptos_images[0])
                self._test("Can load APTOS sample image", True)
                print(f"    Sample APTOS image size: {sample_image.size}")

                # Test basic PIL operations
                resized = sample_image.resize((224, 224))
                self._test("Can resize APTOS image to 224x224", resized.size == (224, 224))

            except Exception as e:
                self._test(f"APTOS image loading failed: {str(e)}", False)
        else:
            self._test("APTOS sample images found", False)

        if sample_eyepacs_images:
            self._test(f"Found EyePACS sample images ({len(sample_eyepacs_images)})", True)

            # Test loading a sample EyePACS image
            try:
                sample_image = Image.open(sample_eyepacs_images[0])
                self._test("Can load EyePACS sample image", True)
                print(f"    Sample EyePACS image size: {sample_image.size}")

                # Test basic PIL operations
                resized = sample_image.resize((299, 299))
                self._test("Can resize EyePACS image to 299x299", resized.size == (299, 299))

            except Exception as e:
                self._test(f"EyePACS image loading failed: {str(e)}", False)
        else:
            self._test("EyePACS sample images found", False)

    def test_aspect_ratio_preservation(self):
        """Test aspect ratio preservation logic"""
        print("\n--- Testing Aspect Ratio Logic ---")

        # Create test images with different aspect ratios
        test_cases = [
            ((400, 200), 224, (224, 112)),  # Wide image -> Pipeline A
            ((200, 400), 224, (112, 224)),  # Tall image -> Pipeline A
            ((600, 300), 299, (299, 149)),  # Wide image -> Pipeline B
            ((300, 600), 299, (149, 299)),  # Tall image -> Pipeline B
        ]

        for original_size, target_size, expected_size in test_cases:
            # Calculate what the resized dimensions should be
            width, height = original_size
            longest_side = max(width, height)
            scale_factor = target_size / longest_side
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            calculated_size = (new_width, new_height)
            test_name = f"Aspect ratio calc: {original_size} -> {target_size} = {expected_size}"
            self._test(test_name, calculated_size == expected_size)

    def _test(self, description, condition):
        """Helper method to run a test and track results"""
        self.total_tests += 1
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {description}")

        if condition:
            self.passed_tests += 1

    def print_test_summary(self):
        """Print summary of all tests"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total tests run: {self.total_tests}")
        print(f"Tests passed: {self.passed_tests}")
        print(f"Tests failed: {self.total_tests - self.passed_tests}")

        if self.total_tests > 0:
            success_rate = (self.passed_tests/self.total_tests)*100
            print(f"Success rate: {success_rate:.1f}%")

            if self.passed_tests == self.total_tests:
                print("\n🎉 ALL BASIC TESTS PASSED! Core data functionality is working.")
            else:
                failed_count = self.total_tests - self.passed_tests
                print(f"\n⚠️  {failed_count} test(s) failed. Please review the issues above.")

        print("="*60)

def main():
    """Run the simplified data pipeline tests"""
    test_suite = SimpleDataTestSuite()
    test_suite.run_basic_tests()
    test_suite.test_aspect_ratio_preservation()

if __name__ == "__main__":
    main()
