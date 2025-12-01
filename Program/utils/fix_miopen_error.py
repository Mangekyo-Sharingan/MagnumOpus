"""
Fix for MIOpen SQLite Database Error
This script clears MIOpen cache and sets environment variables to resolve the error.
"""

import os
import shutil
from pathlib import Path

def fix_miopen_error():
    """
    Fix MIOpen SQLite database error by clearing cache and setting environment variables
    """
    print("=" * 80)
    print("MIOpen Error Fix Tool")
    print("=" * 80)

    # Solution 1: Clear MIOpen cache
    print("\n1. Clearing MIOpen cache...")

    miopen_cache_paths = [
        Path.home() / ".config" / "miopen",
        Path(os.getenv("LOCALAPPDATA", "")) / "AMD" / "MIOpen",
        Path(os.getenv("APPDATA", "")) / "AMD" / "MIOpen",
        Path("C:/Users") / os.getenv("USERNAME", "") / ".miopen"
    ]

    cleared = False
    for cache_path in miopen_cache_paths:
        if cache_path.exists():
            try:
                print(f"   Found cache at: {cache_path}")
                response = input(f"   Delete this cache? (y/n): ").strip().lower()
                if response == 'y':
                    shutil.rmtree(cache_path)
                    print(f"   [OK] Cleared: {cache_path}")
                    cleared = True
            except Exception as e:
                print(f"    Could not clear {cache_path}: {e}")

    if not cleared:
        print("   No MIOpen cache found or nothing cleared")

    # Solution 2: Set environment variables
    print("\n2. Setting MIOpen environment variables...")
    print("   Add these to your system or run before training:\n")

    env_vars = {
        "MIOPEN_DISABLE_CACHE": "1",
        "MIOPEN_FIND_MODE": "NORMAL",
        "MIOPEN_DEBUG_DISABLE_FIND_DB": "1"
    }

    print("   # PowerShell commands:")
    for key, value in env_vars.items():
        print(f"   $env:{key}='{value}'")
        os.environ[key] = value

    print("\n   # Or in Python (before importing torch):")
    for key, value in env_vars.items():
        print(f"   os.environ['{key}'] = '{value}'")

    print("\n[OK] Environment variables set for current session")

    # Solution 3: Create a wrapper script
    print("\n3. Creating wrapper script for main.py...")

    wrapper_content = '''"""
Wrapper script to run main.py with MIOpen error fixes
"""
import os

# Set MIOpen environment variables BEFORE importing torch
os.environ["MIOPEN_DISABLE_CACHE"] = "1"
os.environ["MIOPEN_FIND_MODE"] = "NORMAL"
os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"

print("[OK] MIOpen environment variables set")
print("  - MIOPEN_DISABLE_CACHE = 1")
print("  - MIOPEN_FIND_MODE = NORMAL")
print("  - MIOPEN_DEBUG_DISABLE_FIND_DB = 1")
print()

# Now import and run main
from main import main

if __name__ == "__main__":
    main()
'''

    wrapper_path = Path(__file__).parent / "run_main_with_fix.py"
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_content)

    print(f"   [OK] Created: {wrapper_path}")
    print(f"   Usage: python run_main_with_fix.py")

    print("\n" + "=" * 80)
    print("Fix Applied!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. EITHER run: python run_main_with_fix.py")
    print("2. OR manually set environment variables before running main.py")
    print("3. OR modify main.py to set env vars at the top")
    print("\nIf error persists, try switching to CPU: DeviceManager.force_cpu()")
    print("=" * 80)

if __name__ == "__main__":
    fix_miopen_error()


