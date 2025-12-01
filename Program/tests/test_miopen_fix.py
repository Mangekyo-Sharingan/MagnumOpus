"""
Test script to verify MIOpen fix works before running full training
This clears the corrupted cache and tests basic GPU operations
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to access modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set environment variables FIRST (before importing torch)
os.environ["MIOPEN_DISABLE_CACHE"] = "1"
os.environ["MIOPEN_FIND_MODE"] = "NORMAL"
os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = ""

print("=" * 100)
print(" " * 35 + "MIOPEN FIX TEST")
print("=" * 100)

# Import modules
from modules import DeviceManager
import torch
import torch.nn as nn

def test_miopen_fix():
    """Test if MIOpen fix resolves the SQLite error"""

    # Step 1: Clear MIOpen cache
    print("\n STEP 1: Clearing MIOpen cache...")
    DeviceManager.fix_miopen_cache()

    # Step 2: Check device availability
    print("\n STEP 2: Checking GPU availability...")
    if not torch.cuda.is_available():
        print("[FAIL] CUDA not available! Check your PyTorch installation.")
        return False

    print(f"[OK] CUDA is available!")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Step 3: Test basic tensor operations
    print("\n STEP 3: Testing basic GPU tensor operations...")
    try:
        device = torch.device('cuda')
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        z = torch.matmul(x, y)
        print(f"[OK] Basic tensor operations successful!")
        print(f"   Computed 100x100 matrix multiplication on GPU")
    except Exception as e:
        print(f"[FAIL] Tensor operations failed: {e}")
        return False

    # Step 4: Test convolution (this is where MIOpen errors typically occur)
    print("\n STEP 4: Testing convolution operations (MIOpen test)...")
    try:
        # Create a simple conv layer
        conv = nn.Conv2d(3, 64, kernel_size=3, padding=1).to(device)

        # Create dummy input (batch_size=2, channels=3, height=224, width=224)
        input_tensor = torch.randn(2, 3, 224, 224, device=device)

        # Run forward pass
        output = conv(input_tensor)

        print(f"[OK] Convolution operations successful!")
        print(f"   Input shape: {input_tensor.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"    MIOpen is working correctly!")

    except RuntimeError as e:
        if "MIOpen" in str(e) or "SQLite" in str(e):
            print(f"[FAIL] MIOpen error still present: {e}")
            print("\n[WARNING]  Additional troubleshooting needed:")
            print("   1. Try running this script again (cache may need multiple clears)")
            print("   2. Update AMD ROCm drivers: https://www.amd.com/en/support")
            print("   3. Reinstall PyTorch with ROCm support:")
            print("      pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7")
            return False
        else:
            print(f"[FAIL] Unexpected error: {e}")
            return False

    # Step 5: Test batch norm (another common MIOpen operation)
    print("\n STEP 5: Testing batch normalization...")
    try:
        bn = nn.BatchNorm2d(64).to(device)
        output_bn = bn(output)
        print(f"[OK] Batch normalization successful!")
        print(f"   Output shape: {output_bn.shape}")
    except Exception as e:
        print(f"[FAIL] Batch norm failed: {e}")
        return False

    # Success!
    print("\n" + "=" * 100)
    print(" " * 30 + "[OK] ALL TESTS PASSED!")
    print("=" * 100)
    print("\n MIOpen is working correctly! You can now run your training script.")
    print("   Run: python main.py")
    print("=" * 100)

    return True

if __name__ == "__main__":
    success = test_miopen_fix()

    if not success:
        print("\n" + "=" * 100)
        print(" " * 25 + "[WARNING]  MIOPEN FIX INCOMPLETE")
        print("=" * 100)
        print("\nThe fix may need additional steps. Try:")
        print("  1. Run this script again")
        print("  2. Restart your computer")
        print("  3. Update AMD drivers")
        print("=" * 100)
        sys.exit(1)
    else:
        sys.exit(0)

