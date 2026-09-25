# src/data.py
import pandas as pd
from pathlib import Path
from typing import Callable, Optional


def load_sample(file_path: Path, nrows: int = 10000) -> pd.DataFrame:
    """Reads a small sample of a TSV file safely."""
    return pd.read_csv(file_path, sep="\t", nrows=nrows)


def process_in_chunks(
    file_path: Path,
    chunksize: int = 100000,
    process_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Generates processed chunks without loading the full file into memory."""
    chunks = []
    for chunk in pd.read_csv(file_path, sep="\t", chunksize=chunksize):
        if process_fn:
            chunk = process_fn(chunk)
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)
