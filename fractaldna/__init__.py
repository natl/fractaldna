# type: ignore[attr-defined]
"""Awesome `fractaldna` is a Python cli/package created with https://github.com/TezRomacH/python-package-template"""

import sys

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # for Python<3.8
    import importlib_metadata


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()
