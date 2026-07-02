"""Pytest bootstrap: put the project root on sys.path.

Lets tests do `from core...` / `from agents...` regardless of how pytest is
invoked. pytest auto-imports the nearest conftest.py and adds its directory to
sys.path, so this file living at the repo root is enough.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
