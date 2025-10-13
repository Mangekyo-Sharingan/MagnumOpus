#!/usr/bin/env python3
"""
Master test script for all modules in the diabetic retinopathy classification project.
This script tests each module independently and provides a comprehensive report.

Usage:
    python run_module_tests.py

Returns:
    - Individual test results for each module
    - Overall PASS/FAIL status
    - Detailed error information if any tests fail
"""

import sys
import os
from pathlib import Path
import importlib
from datetime import datetime
import traceback

# Add the modules directory to the path
modules_dir = Path(__file__).parent / "modules"
sys.path.insert(0, str(modules_dir))

class ModuleTester:
    """Class to handle testing of all project modules"""

    def __init__(self):
        self.test_results = {}
        self.modules_to_test = [
            'config',
            'utils',
            'data',
            'models',
            'train',
            'test'
        ]

    def run_all_tests(self):
        """Run tests for all modules"""
        print("=" * 80)
        print("DIABETIC RETINOPATHY PROJECT - MODULE TESTING")
        print("=" * 80)
        print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Testing {len(self.modules_to_test)} modules...")
        print("=" * 80)

        overall_success = True

        for module_name in self.modules_to_test:
            print(f"\n{'='*20} TESTING {module_name.upper()} MODULE {'='*20}")

            try:
                success = self._test_module(module_name)
                self.test_results[module_name] = {
                    'status': 'PASS' if success else 'FAIL',
                    'success': success,
                    'error': None
                }

                if not success:
                    overall_success = False

            except Exception as e:
                print(f"✗ CRITICAL ERROR testing {module_name}: {e}")
                traceback.print_exc()
                self.test_results[module_name] = {
                    'status': 'FAIL',
                    'success': False,
                    'error': str(e)
                }
                overall_success = False

        self._print_summary(overall_success)
        return overall_success

    def _test_module(self, module_name):
        """Test an individual module"""
        try:
            # Import the module
            module = importlib.import_module(module_name)
            print(f"✓ Successfully imported {module_name} module")

            # Find and run the test function
            test_function_name = f"test_{module_name}_module"
            if hasattr(module, test_function_name):
                test_function = getattr(module, test_function_name)
                print(f"✓ Found test function: {test_function_name}")

                # Run the test
                result = test_function()

                if result:
                    print(f"✓ {module_name.upper()} MODULE TEST: PASSED")
                else:
                    print(f"✗ {module_name.upper()} MODULE TEST: FAILED")

                return result
            else:
                print(f"⚠ Warning: No test function found for {module_name}")
                print(f"  Expected function name: {test_function_name}")
                return False

        except ImportError as e:
            print(f"✗ Failed to import {module_name}: {e}")
            return False
        except Exception as e:
            print(f"✗ Error testing {module_name}: {e}")
            traceback.print_exc()
            return False

    def _print_summary(self, overall_success):
        """Print comprehensive test summary"""
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)

        passed_count = sum(1 for result in self.test_results.values() if result['success'])
        total_count = len(self.test_results)

        print(f"Total modules tested: {total_count}")
        print(f"Modules passed: {passed_count}")
        print(f"Modules failed: {total_count - passed_count}")
        print()

        # Detailed results
        print("DETAILED RESULTS:")
        print("-" * 40)
        for module_name, result in self.test_results.items():
            status_symbol = "✓" if result['success'] else "✗"
            status_text = result['status']
            print(f"{status_symbol} {module_name.ljust(10)}: {status_text}")

            if result['error']:
                print(f"    Error: {result['error']}")

        print("\n" + "=" * 80)

        if overall_success:
            print("🎉 ALL MODULES PASSED! 🎉")
            print("The project structure is correctly implemented with PyTorch.")
            print("All modules can be run independently and work as expected.")
        else:
            print("❌ SOME MODULES FAILED")
            print("Please check the detailed error messages above.")
            print("Fix the issues and run the tests again.")

        print("=" * 80)
        print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

def main():
    """Main function to run all module tests"""
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Create tester and run tests
    tester = ModuleTester()
    success = tester.run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
