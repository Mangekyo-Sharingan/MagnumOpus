"""
Fix for MIOpen SQLite Database Error on AMD GPUs
This script clears corrupted MIOpen cache to resolve the error.
"""

import os
import shutil
from pathlib import Path

def fix_miopen_error(disable_cache: bool = True):
    """
    Fix MIOpen SQLite database error.

    If ``disable_cache`` is True (legacy behaviour), clear MIOpen cache directories
    and rely on environment flags that disable the cache.

    If ``disable_cache`` is False, we *do not* delete the cache so that
    MIOpen can reuse previously tuned kernels. This is the recommended
    default for stable training once things work.
    """
    print("=" * 100)
    print(" " * 30 + "MIOPEN ERROR FIX TOOL FOR AMD GPUs")
    print("=" * 100)
    print("\n[WARNING]  ERROR: MIOpen SQLite database: no such column: mode")
    print("\nThis error occurs due to incompatible MIOpen cache database schema.")
    print("FIX: Either clear the corrupted cache once, or keep a working cache.")
    print("=" * 100)

    # Step 1: Optionally clear MIOpen cache
    action = "Clearing" if disable_cache else "Preserving"
    print(f"\n STEP 1: {action} MIOpen cache directories...")
    print("─" * 100)

    miopen_cache_paths = [
        Path.home() / ".cache" / "miopen",
        Path(os.getenv("LOCALAPPDATA", "")) / "AMD" / "MIOpen" if os.getenv("LOCALAPPDATA") else None,
        Path(os.getenv("APPDATA", "")) / "AMD" / "MIOpen" if os.getenv("APPDATA") else None,
        Path("C:/Users") / os.getenv("USERNAME", "") / ".miopen" if os.getenv("USERNAME") else None,
        Path("/tmp/miopen-cache"),
        Path.home() / ".config" / "miopen",
    ]
    miopen_cache_paths = [p for p in miopen_cache_paths if p is not None]

    cleared_count = 0
    preserved_count = 0
    for cache_path in miopen_cache_paths:
        if cache_path.exists():
            if disable_cache:
                try:
                    print(f"    Found cache: {cache_path}")
                    shutil.rmtree(cache_path)
                    print(f"   [OK] Cleared: {cache_path}")
                    cleared_count += 1
                except Exception as e:
                    print(f"   [WARNING]  Could not clear {cache_path}: {e}")
            else:
                print(f"    Found MIOpen cache: {cache_path} (preserved)")
                preserved_count += 1

    if disable_cache:
        if cleared_count > 0:
            print(f"\n[OK] Successfully cleared {cleared_count} MIOpen cache director{'y' if cleared_count == 1 else 'ies'}!")
        else:
            print("\n [INFO]  No MIOpen cache found (may already be clean)")
    else:
        if preserved_count > 0:
            print(f"\n[OK] MIOpen cache preserved in {preserved_count} director{'y' if preserved_count == 1 else 'ies'}.")
        else:
            print("\n [INFO]  No existing MIOpen cache directories found to preserve.")

    # Step 2: Environment variables – favour caching when not disabled
    print("\n STEP 2: Configuring environment variables...")
    print("─" * 100)

    if disable_cache:
        env_vars = {
            "MIOPEN_DISABLE_CACHE": "1",
            "MIOPEN_FIND_MODE": "NORMAL",
            "MIOPEN_DEBUG_DISABLE_FIND_DB": "1",
            "MIOPEN_CUSTOM_CACHE_DIR": "",
        }
        print("\n  Running with cache DISABLED (legacy safe mode).")
    else:
        env_vars = {
            # Explicitly allow cache usage
            "MIOPEN_DISABLE_CACHE": "0",
            "MIOPEN_FIND_MODE": "NORMAL",
            "MIOPEN_DEBUG_DISABLE_FIND_DB": "0",
        }
        print("\n  Running with cache ENABLED to reuse tuned kernels.")

    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"   - {key} = '{value}'")

    print("\n" + "=" * 100)
    print("[OK] Fix script completed!")
    print("=" * 100)

if __name__ == "__main__":
    # Default CLI behaviour keeps legacy safe mode (disable cache).
    # The main training code should call fix_miopen_error(disable_cache=config.miopen_disable_cache)
    # so that your Config controls whether the cache is actually cleared.
    fix_miopen_error(disable_cache=True)
