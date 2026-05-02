"""
Root conftest.py — ensures the project root is always on sys.path so that
`from src.xxx import yyy` works in every test file regardless of how pytest
is invoked.
"""
import sys
import pathlib

# Add the project root (one level above this file) to sys.path
sys.path.insert(0, str(pathlib.Path(__file__).parent))
