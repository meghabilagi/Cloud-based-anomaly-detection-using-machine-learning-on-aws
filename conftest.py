"""
Root conftest.py — adds src/ to sys.path so that modules inside src/
can import each other using bare names (e.g. ``from utils import ...``).
"""
import sys
import os

# Allow src/ modules to import each other without the 'src.' prefix.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
