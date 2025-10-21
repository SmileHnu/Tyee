#!/usr/bin/env python3 -u
"""
A simple script to train models.

This is a convenience wrapper around the `tyee.main` entry point.
"""

import sys
from os.path import abspath, dirname

# Add the project root to sys.path to allow for absolute imports
# This makes it possible to run `python train.py` from the root directory.
sys.path.insert(0, dirname(abspath(__file__)))

from tyee.main import cli_main

if __name__ == "__main__":
    cli_main()
