"""
Quick verification that MIOpen error fix is working
Run this to test if the fix resolves the SQLite database error
"""
import os

# Apply the MIOpen fix
os.environ["MIOPEN_DISABLE_CACHE"] = "1"
os.environ["MIOPEN_FIND_MODE"] = "NORMAL"
os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"

print("=" * 80)
print("MIOpen Error Fix Verification")
print("=" * 80)
print("\n[OK] Environment variables set:")
print(f"  MIOPEN_DISABLE_CACHE = {os.environ.get('MIOPEN_DISABLE_CACHE')}")
print(f"  MIOPEN_FIND_MODE = {os.environ.get('MIOPEN_FIND_MODE')}")
print(f"  MIOPEN_DEBUG_DISABLE_FIND_DB = {os.environ.get('MIOPEN_DEBUG_DISABLE_FIND_DB')}")

print("\n" + "─" * 80)
print("Testing PyTorch GPU availability...")
print("─" * 80)

try:
    import torch
    print(f"[OK] PyTorch version: {torch.__version__}")
    print(f"[OK] CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"[OK] GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"[OK] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        # Test if we can create tensors on GPU
        print("\n" + "─" * 80)
        print("Testing GPU tensor operations...")
        print("─" * 80)

        test_tensor = torch.randn(10, 10).cuda()
        result = test_tensor @ test_tensor.t()
        print("[OK] GPU tensor operations successful")

        # Test a simple convolution (this is where MIOpen errors usually occur)
        print("\n" + "─" * 80)
        print("Testing convolution operations (this triggers MIOpen)...")
        print("─" * 80)

        import torch.nn as nn
        conv = nn.Conv2d(3, 64, kernel_size=3).cuda()
        test_input = torch.randn(1, 3, 224, 224).cuda()

        print("  Running convolution... (this may take a moment on first run)")
        output = conv(test_input)
        print(f"[OK] Convolution successful! Output shape: {output.shape}")

        print("\n" + "=" * 80)
        print("[OK] ALL TESTS PASSED - MIOpen fix is working!")
        print("=" * 80)
        print("\n You can now run your training with confidence!")
        print("   Just use: python main.py")
        print("=" * 80)

    else:
        print("\n[WARNING]  GPU not available - using CPU")
        print("   This is fine, but training will be slower")
        print("   MIOpen error won't occur on CPU")

except Exception as e:
    print(f"\n[FAIL] Error occurred: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure PyTorch is installed: pip install torch torchvision")
    print("2. Check if AMD GPU drivers are installed")
    print("3. Try running: python fix_miopen_error.py")
    import traceback
    traceback.print_exc()

print()

