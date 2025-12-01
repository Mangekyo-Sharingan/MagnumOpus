"""
Quick test to verify transforms are picklable (fixes Windows multiprocessing issue)
"""
import sys
from pathlib import Path

# Add Program to path
sys.path.insert(0, str(Path(__file__).parent / "Program"))

import pickle
from modules.data import CropBlackBordersTransform, ApplyCLAHETransform

def test_transform_pickling():
    """Test that transforms can be pickled (required for Windows multiprocessing)"""

    print("Testing transform pickling...")
    print("=" * 60)

    try:
        # Test CropBlackBordersTransform
        print("\n1. Testing CropBlackBordersTransform...")
        transform1 = CropBlackBordersTransform()
        pickled1 = pickle.dumps(transform1)
        unpickled1 = pickle.loads(pickled1)
        print("   [OK] CropBlackBordersTransform is picklable")

        # Test ApplyCLAHETransform
        print("\n2. Testing ApplyCLAHETransform...")
        transform2 = ApplyCLAHETransform()
        pickled2 = pickle.dumps(transform2)
        unpickled2 = pickle.loads(pickled2)
        print("   [OK] ApplyCLAHETransform is picklable")

        print("\n" + "=" * 60)
        print("[OK] ALL TESTS PASSED!")
        print("=" * 60)
        print("\nTransforms are now picklable and can be used with")
        print("multiprocessing DataLoaders on Windows.")

        return True

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_transform_pickling()
    sys.exit(0 if success else 1)

