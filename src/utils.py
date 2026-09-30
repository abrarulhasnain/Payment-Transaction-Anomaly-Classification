"""
utils.py
========
Small shared helpers used across the project: logging setup, JSON IO, and a
guard that gives a clear, honest error instead of silently fabricating
results when the real dataset is missing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that writes to stdout."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def save_json(data: dict, path: Path) -> None:
    """Save a dict as pretty-printed JSON, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> Any:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class DatasetNotFoundError(FileNotFoundError):
    """
    Raised when the real dataset has not been placed in data/raw/ yet.

    This project deliberately refuses to fabricate results. Any script that
    needs the real data will raise this error with clear instructions rather
    than silently generating a synthetic dataset and presenting its output
    as if it were real.
    """

    def __init__(self, expected_path: Path):
        message = (
            f"\n\nDataset not found at: {expected_path}\n"
            "This project does not fabricate results. To proceed:\n"
            "  1. Obtain a transaction anomaly/fraud dataset (see data/README.md\n"
            "     for a recommended free/open-source option).\n"
            "  2. Place the CSV file at the path above (or update\n"
            "     src/config.py -> RAW_DATA_FILENAME / RAW_DATA_PATH).\n"
            "  3. Update src/config.py -> FEATURE_CONFIG with the dataset's\n"
            "     actual column names.\n"
            "  4. Re-run this script.\n"
        )
        super().__init__(message)
