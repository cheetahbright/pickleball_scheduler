#!/usr/bin/env python3
"""Hosted Streamlit entry point for the pickleball scheduler deployment."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from src.main_app import main


main()
