"""
Test script to verify MIOpen fix toggles work correctly
This script helps you test different MIOpen fix configurations without running full training
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to access modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules import Config, DeviceManager

def test_configuration(config_name, env_vars=None):
    """Test a specific MIOpen configuration"""
    print("\n" + "=" * 80)
    print(f"Testing: {config_name}")
    print("=" * 80)

    # Set environment variables if provided
    if env_vars:
        print(f"\nEnvironment variables set:")
        for key, value in env_vars.items():
            os.environ[key] = value
            print(f"  {key} = {value}")

    # Create config
    config = Config(use_logger=False)

    # Display configuration
    print(f"\nConfiguration loaded:")
    print(f"  enable_miopen_fix: {config.enable_miopen_fix}")
    print(f"  miopen_disable_cache: {config.miopen_disable_cache}")
    if config.miopen_disable_cache:
        print(f"      Mode: COMPATIBILITY (cache disabled - slower but stable)")
    else:
        print(f"     Mode: PERFORMANCE (cache enabled - faster)")
    print(f"  MIOpen fix sources:")
    for source, enabled in config.miopen_fix_sources.items():
        status = "[OK] ENABLED" if enabled else "[FAIL] DISABLED"
        print(f"    - {source}: {status}")

    # Test applying fixes (without actually running)
    print(f"\nApplying MIOpen fixes...")
    if config.enable_miopen_fix:
        DeviceManager.apply_miopen_fixes(config)
    else:
        print("[INFO]  MIOpen fixes disabled via configuration")

    # Clean up environment variables
    if env_vars:
        for key in env_vars.keys():
            os.environ.pop(key, None)

    print("=" * 80)


def main():
    """Run various MIOpen fix configuration tests"""
    print("\n" + "" * 40)
    print("  MIOPEN FIX TOGGLE TEST SUITE")
    print("" * 40)

    # Test 1: Default configuration (PERFORMANCE MODE - cache enabled)
    test_configuration(
        "DEFAULT CONFIGURATION (PERFORMANCE MODE)",
        env_vars=None
    )

    # Test 2: Compatibility mode (cache disabled for stability)
    test_configuration(
        "COMPATIBILITY MODE (CACHE DISABLED)",
        env_vars={"MIOPEN_DISABLE_CACHE_OPT": "1"}
    )

    # Test 3: All fixes disabled
    test_configuration(
        "ALL FIXES DISABLED",
        env_vars={"ENABLE_MIOPEN_FIX": "0"}
    )

    # Test 4: Enable fix_miopen_amd.py
    test_configuration(
        "WITH fix_miopen_amd.py ENABLED",
        env_vars={"MIOPEN_FIX_AMD_SCRIPT": "1"}
    )

    # Test 5: Enable fix_miopen_error.py
    test_configuration(
        "WITH fix_miopen_error.py ENABLED",
        env_vars={"MIOPEN_FIX_LEGACY_SCRIPT": "1"}
    )

    # Test 6: Enable both external scripts
    test_configuration(
        "WITH BOTH EXTERNAL SCRIPTS ENABLED",
        env_vars={
            "MIOPEN_FIX_AMD_SCRIPT": "1",
            "MIOPEN_FIX_LEGACY_SCRIPT": "1"
        }
    )

    # Test 7: Only device_manager fix disabled
    test_configuration(
        "DEVICE_MANAGER FIX DISABLED",
        env_vars={"MIOPEN_FIX_DEVICE_MANAGER": "0"}
    )

    print("\n" + "=" * 80)
    print("[OK] All configuration tests completed!")
    print("=" * 80)

    # Print usage instructions
    print("\n" + "" * 40)
    print("  USAGE INSTRUCTIONS")
    print("" * 40)
    print("""
To control MIOpen fixes in your training, use environment variables:

 PERFORMANCE MODE (DEFAULT - RECOMMENDED):
   python main.py
   - MIOpen cache ENABLED for maximum performance
   - Cache is cleared on startup to avoid corruption
   - Best choice if you're not experiencing MIOpen errors

 COMPATIBILITY MODE (if you get MIOpen errors):
   $env:MIOPEN_DISABLE_CACHE_OPT="1"; python main.py
   - MIOpen cache DISABLED (slower but more stable)
   - Use this if you see "no such column: mode" errors

Other toggles:

1. DISABLE ALL FIXES:
   $env:ENABLE_MIOPEN_FIX="0"; python main.py

2. ENABLE fix_miopen_amd.py:
   $env:MIOPEN_FIX_AMD_SCRIPT="1"; python main.py

3. ENABLE fix_miopen_error.py:
   $env:MIOPEN_FIX_LEGACY_SCRIPT="1"; python main.py

4. ENABLE BOTH EXTERNAL SCRIPTS:
   $env:MIOPEN_FIX_AMD_SCRIPT="1"; $env:MIOPEN_FIX_LEGACY_SCRIPT="1"; python main.py

5. DISABLE ONLY DeviceManager fix:
   $env:MIOPEN_FIX_DEVICE_MANAGER="0"; python main.py

Or modify config.py directly (lines 78-99) to change default values.
""")
    print("=" * 80)


if __name__ == "__main__":
    main()

