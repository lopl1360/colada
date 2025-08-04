"""Utilities for managing trading symbols."""
import os


def load_symbols():
    """Return a list of symbols defined in ``symbols.txt``.

    The file is expected to contain one symbol per line and is located in the
    same directory as this module.
    """
    symbols_file = os.path.join(os.path.dirname(__file__), "symbols.txt")
    with open(symbols_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
