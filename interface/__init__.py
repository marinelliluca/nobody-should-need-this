"""Gradio interfaces for the `rag` and `scraper` packages.

Two independently-launched apps share a small set of helpers in
`interface.common`:

    python -m interface.rag_app
    python -m interface.scraper_app
    # or, for hot-reload during development:
    gradio interface/rag_app.py

Importing this package disables Gradio analytics for every app. This is set
here, before any submodule imports `gradio`.
"""
from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
