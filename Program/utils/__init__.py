"""
Utils package for MagnumOpus project
Contains utility modules for logging, output management, and MIOpen fixes
"""

from .excel_logger import ExcelLogger

__all__ = [
    'ExcelLogger',
]

# MIOpen fix utilities are available as separate modules:
# - utils.fix_miopen_amd
# - utils.fix_miopen_error
# - utils.verify_miopen_fix
# - utils.miopen_quick_ref

