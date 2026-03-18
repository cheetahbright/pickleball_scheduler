#!/usr/bin/env python3
"""Legacy enhanced GUI entry point.

The original multi-algorithm UI has been folded into the main Streamlit app.
Keep this wrapper so older launch paths and docs still work.
"""

from src.main_app import main

if __name__ == "__main__":
    main()
