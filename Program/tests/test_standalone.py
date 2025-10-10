"""
Standalone test script for data validation
This bypasses module imports to avoid TensorFlow compatibility issues
"""
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

def test_data_structure():
    """Test the basic data structure and files"""
    print("="*60)
    print("STANDALONE DATA VALIDATION TESTS")
    print("="*60)

    passed_tests = 0
    total_tests = 0

    def test(description, condition):
        nonlocal passed_tests, total_tests
        total_tests += 1
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {description}")
        if condition:
            passed_tests += 1
        return condition

    # Define paths directly
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "Data"
    aptos_dir = data_dir / "Aptos"
    eyepacs_dir = data_dir / "EyePacs"

    print("\n--- Testing Directory Structure ---")
    test("Data directory exists", data_dir.exists())
    test("APTOS directory exists", aptos_dir.exists())
    test("EyePACS directory exists", eyepacs_dir.exists())

    # Test APTOS files
    aptos_train_csv = aptos_dir / "train.csv"
    aptos_test_csv = aptos_dir / "test.csv"
    aptos_train_images = aptos_dir / "train_images"
    aptos_test_images = aptos_dir / "test_images"

    print("\n--- Testing APTOS Dataset ---")
    test("APTOS train.csv exists", aptos_train_csv.exists())
    test("APTOS test.csv exists", aptos_test_csv.exists())
    test("APTOS train_images directory exists", aptos_train_images.exists())
    test("APTOS test_images directory exists", aptos_test_images.exists())

    if aptos_train_csv.exists():
        try:
            aptos_df = pd.read_csv(aptos_train_csv)
            test(f"APTOS train.csv loaded ({len(aptos_df)} samples)", len(aptos_df) > 0)

            required_cols = ['id_code', 'diagnosis']
            test("APTOS has required columns", all(col in aptos_df.columns for col in required_cols))

            valid_diagnoses = aptos_df['diagnosis'].isin([0, 1, 2, 3, 4]).all()
            test("APTOS diagnosis values valid (0-4)", valid_diagnoses)

            print("    APTOS diagnosis distribution:")
            for diag, count in aptos_df['diagnosis'].value_counts().sort_index().items():
                percentage = (count / len(aptos_df)) * 100
                print(f"      Level {diag}: {count} ({percentage:.1f}%)")

        except Exception as e:
            test(f"APTOS CSV loading failed: {str(e)}", False)

    # Test EyePACS files
    eyepacs_train_csv = eyepacs_dir / "trainLabels.csv" / "trainLabels.csv"
    eyepacs_train_images = eyepacs_dir / "train"
    eyepacs_test_images = eyepacs_dir / "test"

    print("\n--- Testing EyePACS Dataset ---")
    test("EyePACS trainLabels.csv exists", eyepacs_train_csv.exists())
    test("EyePACS train directory exists", eyepacs_train_images.exists())
    test("EyePACS test directory exists", eyepacs_test_images.exists())

    if eyepacs_train_csv.exists():
        try:
            eyepacs_df = pd.read_csv(eyepacs_train_csv)
            test(f"EyePACS trainLabels.csv loaded ({len(eyepacs_df)} samples)", len(eyepacs_df) > 0)

            required_cols = ['image', 'level']
            test("EyePACS has required columns", all(col in eyepacs_df.columns for col in required_cols))

            valid_levels = eyepacs_df['level'].isin([0, 1, 2, 3, 4]).all()
            test("EyePACS level values valid (0-4)", valid_levels)

            print("    EyePACS level distribution:")
            for level, count in eyepacs_df['level'].value_counts().sort_index().items():
                percentage = (count / len(eyepacs_df)) * 100
                print(f"      Level {level}: {count} ({percentage:.1f}%)")

        except Exception as e:
            test(f"EyePACS CSV loading failed: {str(e)}", False)

    # Test sample images
    print("\n--- Testing Sample Images ---")

    # APTOS sample images
    if aptos_train_images.exists():
        aptos_samples = list(aptos_train_images.glob("*.png"))[:5]
        test(f"Found APTOS sample images ({len(aptos_samples)})", len(aptos_samples) > 0)

        if aptos_samples:
            try:
                sample_img = Image.open(aptos_samples[0])
                test("Can load APTOS sample image", True)
                print(f"    Sample APTOS image: {aptos_samples[0].name}, size: {sample_img.size}")

                # Test resizing for Pipeline A (224x224)
                resized_a = sample_img.resize((224, 224))
                test("Can resize APTOS to 224x224", resized_a.size == (224, 224))

            except Exception as e:
                test(f"APTOS image loading failed: {str(e)}", False)

    # EyePACS sample images
    if eyepacs_train_images.exists():
        eyepacs_samples = list(eyepacs_train_images.glob("*.jpeg"))[:5]
        test(f"Found EyePACS sample images ({len(eyepacs_samples)})", len(eyepacs_samples) > 0)

        if eyepacs_samples:
            try:
                sample_img = Image.open(eyepacs_samples[0])
                test("Can load EyePACS sample image", True)
                print(f"    Sample EyePACS image: {eyepacs_samples[0].name}, size: {sample_img.size}")

                # Test resizing for Pipeline B (299x299)
                resized_b = sample_img.resize((299, 299))
                test("Can resize EyePACS to 299x299", resized_b.size == (299, 299))

            except Exception as e:
                test(f"EyePACS image loading failed: {str(e)}", False)

    # Test aspect ratio preservation logic
    print("\n--- Testing Aspect Ratio Logic ---")

    test_cases = [
        ((400, 200), 224, (224, 112)),  # Wide -> Pipeline A
        ((200, 400), 224, (112, 224)),  # Tall -> Pipeline A
        ((600, 300), 299, (299, 149)),  # Wide -> Pipeline B
        ((300, 600), 299, (149, 299)),  # Tall -> Pipeline B
    ]

    for original_size, target_size, expected_size in test_cases:
        width, height = original_size
        longest_side = max(width, height)
        scale_factor = target_size / longest_side
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        calculated_size = (new_width, new_height)
        test_name = f"Aspect ratio: {original_size} -> {expected_size}"
        test(test_name, calculated_size == expected_size)

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")

    if total_tests > 0:
        success_rate = (passed_tests/total_tests)*100
        print(f"Success rate: {success_rate:.1f}%")

        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! Data structure is ready for training.")
        else:
            print(f"\n⚠️ {total_tests - passed_tests} test(s) failed.")

    print("="*60)

if __name__ == "__main__":
    test_data_structure()
