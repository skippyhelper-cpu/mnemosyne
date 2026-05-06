"""Mnemosyne CLI entry point — absolute imports for PyInstaller compatibility."""
import sys
import os

# Ensure package is on path
if __name__.startswith('mnemosyne'):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnemosyne.queries.cli import main
sys.exit(main())
