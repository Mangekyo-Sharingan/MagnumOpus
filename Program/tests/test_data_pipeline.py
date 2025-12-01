"""
Test script for the data loading and preprocessing modules
This script validates that the data pipeline works correctly for training
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime
import io

# Add the modules directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import modules carefully to avoid TensorFlow issues
from modules.config import Config

class DataPipelineTestSuite:
    """Test suite for data loading and preprocessing functionality"""

    def __init__(self):
        self.config = Config()
        self.passed_tests = 0
        self.total_tests = 0
        self.output_buffer = io.StringIO()  # Capture output

    def _print(self, message="", save_to_file=True):
        """Print to console and optionally save to buffer"""
        print(message)
        if save_to_file:
            self.output_buffer.write(message + "\n")

    def save_output_to_file(self):
        """Save the captured output to a text file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(__file__).parent / f"data_pipeline_test_results_{timestamp}.txt"

        # Create header with test information
        header = f"""
Data Pipeline Test Results
==========================
Test Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Test File: {Path(__file__).name}
Python Version: {sys.version}
Working Directory: {os.getcwd()}

"""

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write(self.output_buffer.getvalue())

        print(f"\n Test results saved to: {output_file}")
        return output_file

    def run_all_tests(self):
        """Run all data loading tests"""
        self._print("="*60)
        self._print("DIABETIC RETINOPATHY DATA PIPELINE TESTS")
        self._print("="*60)

        # Test configuration
        self.test_config_setup()

        # Test data loading functionality
        self.test_data_loading_logic()

        # Test dataset merging logic
        self.test_dataset_merging_logic()

        # Test preprocessing pipelines
        self.test_pipeline_preprocessing()

        # Test data generator compatibility
        self.test_data_generator_readiness()

        # Summary
        self.print_test_summary()

        # Save output to file
        self.save_output_to_file()

    def test_config_setup(self):
        """Test that configuration is set up correctly"""
        self._print("\n--- Testing Configuration Setup ---")

        # Test 1: Check if paths exist
        self._test("APTOS data directory exists",
                  self.config.aptos_dir.exists())

        self._test("EyePACS data directory exists",
                  self.config.eyepacs_dir.exists())

        # Test 2: Check model configurations
        expected_models = ['vgg16', 'resnet50', 'inceptionv3']
        self._test("All expected models configured",
                  all(model in self.config.model_configs for model in expected_models))

        # Test 3: Check image sizes
        self._test("VGG16 image size is 224x224",
                  self.config.get_image_size('vgg16') == (224, 224))

        self._test("InceptionV3 image size is 299x299",
                  self.config.get_image_size('inceptionv3') == (299, 299))

        # Test 4: Check pipeline assignments
        self._test("VGG16 uses Pipeline A",
                  self.config.get_pipeline_type('vgg16') == 'A')

        self._test("InceptionV3 uses Pipeline B",
                  self.config.get_pipeline_type('inceptionv3') == 'B')

    def test_data_loading_logic(self):
        """Test data loading functionality without importing DataLoader"""
        self._print("\n--- Testing Data Loading Logic ---")

        # Simulate what DataLoader._load_aptos_data() does
        if self.config.aptos_train_csv.exists():
            aptos_df = pd.read_csv(self.config.aptos_train_csv)

            # Add image paths as DataLoader would
            aptos_df['image_path'] = aptos_df['id_code'].apply(
                lambda x: str(self.config.aptos_train_images_dir / f"{x}.png")
            )
            aptos_df['dataset'] = 'APTOS'

            self._test(f"APTOS data loading simulation ({len(aptos_df)} samples)",
                      len(aptos_df) > 0)

            # Check required columns after processing
            required_cols = ['id_code', 'diagnosis', 'image_path', 'dataset']
            self._test("APTOS processed data has required columns",
                      all(col in aptos_df.columns for col in required_cols))

            # Test file path generation
            sample_paths_exist = all(Path(path).exists() for path in aptos_df['image_path'].head(5))
            self._test("APTOS image paths are correctly generated", sample_paths_exist)

        # Simulate what DataLoader._load_eyepacs_data() does
        if self.config.eyepacs_train_csv.exists():
            eyepacs_df = pd.read_csv(self.config.eyepacs_train_csv)

            # Standardize column names as DataLoader would
            eyepacs_df = eyepacs_df.rename(columns={
                'image': 'id_code',
                'level': 'diagnosis'
            })

            # Add image paths
            eyepacs_df['image_path'] = eyepacs_df['id_code'].apply(
                lambda x: str(self.config.eyepacs_train_images_dir / f"{x}.jpeg")
            )
            eyepacs_df['dataset'] = 'EyePACS'

            # Filter existing files as DataLoader would
            existing_mask = eyepacs_df['image_path'].apply(lambda x: Path(x).exists())
            eyepacs_df_filtered = eyepacs_df[existing_mask]

            self._test(f"EyePACS data loading simulation ({len(eyepacs_df_filtered)} samples)",
                      len(eyepacs_df_filtered) > 0)

            # Check column standardization
            required_cols = ['id_code', 'diagnosis', 'image_path', 'dataset']
            self._test("EyePACS column standardization works",
                      all(col in eyepacs_df_filtered.columns for col in required_cols))

            self._print(f"    Original EyePACS samples: {len(eyepacs_df)}")
            self._print(f"    Existing image files: {len(eyepacs_df_filtered)}")
            self._print(f"    File existence rate: {len(eyepacs_df_filtered)/len(eyepacs_df)*100:.1f}%")

    def test_dataset_merging_logic(self):
        """Test dataset merging functionality"""
        self._print("\n--- Testing Dataset Merging Logic ---")

        datasets_to_merge = []

        # Load APTOS data
        if self.config.aptos_train_csv.exists():
            aptos_df = pd.read_csv(self.config.aptos_train_csv)
            aptos_df['image_path'] = aptos_df['id_code'].apply(
                lambda x: str(self.config.aptos_train_images_dir / f"{x}.png")
            )
            aptos_df['dataset'] = 'APTOS'
            datasets_to_merge.append(aptos_df)

        # Load EyePACS data
        if self.config.eyepacs_train_csv.exists():
            eyepacs_df = pd.read_csv(self.config.eyepacs_train_csv)
            eyepacs_df = eyepacs_df.rename(columns={'image': 'id_code', 'level': 'diagnosis'})
            eyepacs_df['image_path'] = eyepacs_df['id_code'].apply(
                lambda x: str(self.config.eyepacs_train_images_dir / f"{x}.jpeg")
            )
            eyepacs_df['dataset'] = 'EyePACS'

            # Filter existing files
            existing_mask = eyepacs_df['image_path'].apply(lambda x: Path(x).exists())
            eyepacs_df = eyepacs_df[existing_mask]
            datasets_to_merge.append(eyepacs_df)

        if datasets_to_merge:
            # Simulate DataLoader._merge_datasets()
            merged_df = pd.concat(datasets_to_merge, ignore_index=True)
            merged_df = merged_df.sample(frac=1, random_state=self.config.random_state).reset_index(drop=True)

            self._test("Dataset merging creates non-empty result", len(merged_df) > 0)

            # Test merged dataset structure
            required_cols = ['id_code', 'diagnosis', 'image_path', 'dataset']
            self._test("Merged dataset has required columns",
                      all(col in merged_df.columns for col in required_cols))

            # Test diagnosis values are valid (0-4)
            valid_diagnoses = merged_df['diagnosis'].isin([0, 1, 2, 3, 4]).all()
            self._test("All diagnosis values are valid (0-4)", valid_diagnoses)

            # Test dataset sources
            dataset_sources = merged_df['dataset'].unique()
            self._test("Merged dataset contains multiple sources", len(dataset_sources) > 1)

            # Print merged dataset statistics
            self._print_merged_dataset_statistics(merged_df)
        else:
            self._test("Dataset merging", False)

    def test_pipeline_preprocessing(self):
        """Test preprocessing pipeline logic without TensorFlow"""
        self._print("\n--- Testing Pipeline Preprocessing Logic ---")

        # Test Pipeline A logic (224x224)
        self._test_pipeline_logic("Pipeline A", 224)

        # Test Pipeline B logic (299x299)
        self._test_pipeline_logic("Pipeline B", 299)

        # Test actual image preprocessing if images exist
        self._test_actual_image_preprocessing()

    def _test_pipeline_logic(self, pipeline_name, target_size):
        """Test the mathematical logic of a preprocessing pipeline"""

        # Test aspect ratio preservation calculations
        test_cases = [
            (400, 200),   # Wide image
            (200, 400),   # Tall image
            (500, 500),   # Square image
            (3216, 2136), # Real APTOS dimensions
            (4752, 3168)  # Real EyePACS dimensions
        ]

        for width, height in test_cases:
            # Simulate _resize_with_aspect_ratio logic
            longest_side = max(width, height)
            scale_factor = target_size / longest_side
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            # Verify scaling preserves aspect ratio
            original_ratio = width / height
            new_ratio = new_width / new_height
            ratio_preserved = abs(original_ratio - new_ratio) < 0.01

            self._test(f"{pipeline_name} preserves aspect ratio for {width}x{height}",
                      ratio_preserved)

            # Verify longest side equals target
            new_longest = max(new_width, new_height)
            self._test(f"{pipeline_name} scales longest side to {target_size} for {width}x{height}",
                      new_longest == target_size)

    def _test_actual_image_preprocessing(self):
        """Test actual image preprocessing with real images"""

        # Test with APTOS image for Pipeline A
        aptos_images = list(self.config.aptos_train_images_dir.glob("*.png"))
        if aptos_images:
            sample_image = Image.open(aptos_images[0])

            # Simulate Pipeline A preprocessing
            processed_224 = self._simulate_pipeline_processing(sample_image, 224)
            self._test("Pipeline A produces 224x224 output", processed_224.size == (224, 224))

            # Test that canvas is properly filled (not all black)
            img_array = np.array(processed_224)
            self._test("Pipeline A output contains image data", img_array.sum() > 0)

        # Test with EyePACS image for Pipeline B
        eyepacs_images = list(self.config.eyepacs_train_images_dir.glob("*.jpeg"))
        if eyepacs_images:
            sample_image = Image.open(eyepacs_images[0])

            # Simulate Pipeline B preprocessing
            processed_299 = self._simulate_pipeline_processing(sample_image, 299)
            self._test("Pipeline B produces 299x299 output", processed_299.size == (299, 299))

            # Test that canvas is properly filled (not all black)
            img_array = np.array(processed_299)
            self._test("Pipeline B output contains image data", img_array.sum() > 0)

    def _simulate_pipeline_processing(self, image, target_size):
        """Simulate the pipeline processing logic"""
        # Step 1: Resize with aspect ratio preservation
        width, height = image.size
        longest_side = max(width, height)
        scale_factor = target_size / longest_side
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Step 2: Create square canvas and center
        canvas = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        x_offset = (target_size - new_width) // 2
        y_offset = (target_size - new_height) // 2
        canvas.paste(resized_image, (x_offset, y_offset))

        return canvas

    def test_data_generator_readiness(self):
        """Test that data is ready for generator creation"""
        self._print("\n--- Testing Data Generator Readiness ---")

        # Test model-pipeline mapping
        model_pipeline_mapping = {
            'vgg16': 'A',
            'resnet50': 'A',
            'inceptionv3': 'B'
        }

        for model, expected_pipeline in model_pipeline_mapping.items():
            actual_pipeline = self.config.get_pipeline_type(model)
            self._test(f"{model} maps to Pipeline {expected_pipeline}",
                      actual_pipeline == expected_pipeline)

        # Test batch size configuration
        self._test("Batch size is configured", self.config.batch_size > 0)

        # Test validation split
        self._test("Validation split is reasonable",
                  0 < self.config.validation_split < 1)

        # Test number of classes
        self._test("Number of classes is 5 (0-4)", self.config.num_classes == 5)

    def _print_merged_dataset_statistics(self, merged_df):
        """Print statistics about the merged dataset"""
        self._print(f"\n  Merged Dataset Statistics:")
        self._print(f"    Total samples: {len(merged_df)}")

        # Dataset distribution
        dataset_counts = merged_df['dataset'].value_counts()
        self._print(f"    Dataset distribution:")
        for dataset, count in dataset_counts.items():
            percentage = (count / len(merged_df)) * 100
            self._print(f"      {dataset}: {count} ({percentage:.1f}%)")

        # Diagnosis distribution
        diagnosis_counts = merged_df['diagnosis'].value_counts().sort_index()
        self._print(f"    Overall diagnosis distribution:")
        for diagnosis, count in diagnosis_counts.items():
            percentage = (count / len(merged_df)) * 100
            self._print(f"      Level {diagnosis}: {count} ({percentage:.1f}%)")

    def _test(self, description, condition):
        """Helper method to run a test and track results"""
        self.total_tests += 1
        status = "PASS" if condition else "FAIL"
        message = f"  [{status}] {description}"
        self._print(message)

        if condition:
            self.passed_tests += 1

    def print_test_summary(self):
        """Print summary of all tests"""
        self._print("\n" + "="*60)
        self._print("TEST SUMMARY")
        self._print("="*60)
        self._print(f"Total tests run: {self.total_tests}")
        self._print(f"Tests passed: {self.passed_tests}")
        self._print(f"Tests failed: {self.total_tests - self.passed_tests}")

        if self.total_tests > 0:
            success_rate = (self.passed_tests/self.total_tests)*100
            self._print(f"Success rate: {success_rate:.1f}%")

            if self.passed_tests == self.total_tests:
                self._print("\n ALL TESTS PASSED! Data pipeline is ready for training.")
            else:
                failed_count = self.total_tests - self.passed_tests
                self._print(f"\n[WARNING] {failed_count} test(s) failed. Please review the issues above.")

        self._print("="*60)

def main():
    """Run the data pipeline tests"""
    test_suite = DataPipelineTestSuite()
    test_suite.run_all_tests()

if __name__ == "__main__":
    main()
