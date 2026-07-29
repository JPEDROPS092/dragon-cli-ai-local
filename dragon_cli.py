#!/usr/bin/env python3
"""Backwards-compatible entry point: `python dragon_cli.py` still works
without installing the package. See src/dragon_cli/ for the actual code."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dragon_cli.cli import main

if __name__ == "__main__":
    main()
