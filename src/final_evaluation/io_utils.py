"""File and NumPy serialization helpers for public SDM evidence."""

import hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def fingerprint(path):
    """Return SHA256/size/mtime for path using streamed reads, without modifying it."""
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(sha256=digest.hexdigest(), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def json_default(value):
    """Convert NumPy scalar/array or path value for evidence JSON; return plain value."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)
