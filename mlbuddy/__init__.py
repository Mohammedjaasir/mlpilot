import sys

# Reconfigure stdout/stderr to UTF-8 on Windows to support emojis in CLI/notebook prints
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from .auto.data import load, load_csv
from .auto.trainer import train, compare

__version__ = "0.1.3"
__all__ = ["load", "load_csv", "train", "compare"]