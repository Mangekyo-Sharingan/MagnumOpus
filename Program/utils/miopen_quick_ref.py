"""
QUICK REFERENCE: MIOpen Fix Toggles
====================================

CURRENT DEFAULT STATUS:
   PERFORMANCE MODE                  - ENABLED (cache enabled for speed)
  [OK] DeviceManager fix (built-in)     - ENABLED (clears cache on startup)
  [FAIL] fix_miopen_amd.py (external)     - DISABLED
  [FAIL] fix_miopen_error.py (external)   - DISABLED (has bug, don't use)

POWERSHELL COMMANDS FOR TESTING:
---------------------------------

Default - PERFORMANCE MODE (recommended):
  python main.py
  → Cache ENABLED for maximum performance
  → Cache cleared on startup to avoid corruption

Switch to COMPATIBILITY MODE (if you get MIOpen errors):
  $env:MIOPEN_DISABLE_CACHE_OPT="1"; python main.py
  → Cache DISABLED (slower but more stable)
  → Use if you see "no such column: mode" errors

Test if fixes are needed at all:
  $env:ENABLE_MIOPEN_FIX="0"; python main.py

Enable AMD script:
  $env:MIOPEN_FIX_AMD_SCRIPT="1"; python main.py

Test all configurations:
  python test_miopen_toggles.py

For full documentation, see: READMEs/MIOPEN_TOGGLES_GUIDE.md
"""
print(__doc__)

