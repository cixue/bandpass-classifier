"""Utility functions for the bandpass classifier.

This module provides common utilities for caching, category conversion, environment
variable management, path resolution, and missing value broadcasting across pandas DataFrames.

Environment Variables:
    BANDPASS_ENABLE_CACHE: Set to '1', 'true', or 'yes' to enable joblib disk caching (default: disabled).
    BANDPASS_CACHE_DIR: Custom path for the joblib cache directory (default: '.cache').
    BANDPASS_MAX_WORKERS: Maximum worker process count for parallel feature extraction (default: auto/all cores).
    BANDPASS_DATA_DIR: Base directory path for training data patterns (default: none / paths as configured).
"""

import functools
import os
from pathlib import Path

import pandas as pd
from joblib import Memory

ENV_VARS: dict[str, dict[str, str]] = {
    "BANDPASS_ENABLE_CACHE": {
        "description": "Enable joblib disk caching for computationally intensive feature extractors.",
        "default": "0 (disabled)",
    },
    "BANDPASS_CACHE_DIR": {
        "description": "Directory path where joblib cache artifacts will be stored when caching is enabled.",
        "default": ".cache",
    },
    "BANDPASS_MAX_WORKERS": {
        "description": "Number of worker processes to use for parallel feature extraction in process_map.",
        "default": "None (auto/all cores)",
    },
    "BANDPASS_DATA_DIR": {
        "description": "Base directory for training data patterns (supports absolute and relative paths).",
        "default": "None (uses paths as configured in config.toml)",
    },
}


def is_cache_enabled() -> bool:
    """Checks whether joblib caching is enabled via environment variable.

    Returns:
        bool: True if caching is enabled via BANDPASS_ENABLE_CACHE, False otherwise (default).
    """
    return os.environ.get("BANDPASS_ENABLE_CACHE", "").lower() in ("1", "true", "yes")


def get_cache_location() -> str:
    """Gets the directory location for joblib caching.

    Returns:
        str: The cache directory path specified by BANDPASS_CACHE_DIR, defaulting to '.cache'.
    """
    return os.environ.get(
        "BANDPASS_CACHE_DIR", ENV_VARS["BANDPASS_CACHE_DIR"]["default"]
    )


def get_max_workers() -> int | None:
    """Gets the configured maximum worker count for parallel processing.

    Returns:
        Optional[int]: The integer worker count from BANDPASS_MAX_WORKERS, or None if unset/invalid.
    """
    val = os.environ.get("BANDPASS_MAX_WORKERS")
    if val is not None and val.isdigit():
        return int(val)
    return None


def get_data_dir() -> Path | None:
    """Gets the configured base directory for training data.

    Returns:
        Optional[Path]: The Path object for the base data directory specified by
            BANDPASS_DATA_DIR (or BANDPASS_TRAINING_DATA_DIR), or None if unset.
    """
    val = os.environ.get("BANDPASS_DATA_DIR") or os.environ.get(
        "BANDPASS_TRAINING_DATA_DIR"
    )
    if val:
        return Path(val)
    return None


def get_env_vars_help_epilog() -> str:
    """Generates formatted help text describing supported environment variables.

    Returns:
        str: Formatted epilog string suitable for argparse help outputs.
    """
    lines = ["environment variables:"]
    for name, meta in ENV_VARS.items():
        desc = meta["description"]
        default = meta["default"]
        lines.append(f"  {name:<24} {desc} (default: {default})")
    return "\n".join(lines)


@functools.cache
def memory() -> Memory:
    """Gets a cached joblib Memory object for caching computations.

    Returns:
        Memory: A joblib Memory object initialized with the cache directory
            (from BANDPASS_CACHE_DIR or default '.cache') if enabled via BANDPASS_ENABLE_CACHE,
            or with location=None (default).
    """
    if is_cache_enabled():
        return Memory(location=get_cache_location(), verbose=0)
    return Memory(location=None, verbose=0)


def convert_to_category(
    df: pd.DataFrame, column: str, categories: list[str] | None = None
) -> list[str] | None:
    """Converts a DataFrame column to a categorical code representation.

    If categories are provided, they are set as the categories for the column.
    Otherwise, the existing unique values in the column are used to define the categories.
    In both cases, the column is replaced with its integer category codes.

    Args:
        df: The pandas DataFrame to modify in-place.
        column: The name of the column to convert.
        categories: An optional list of categories to impose on the column.

    Returns:
        Optional[List[str]]: The list of categories if new categories were inferred,
            otherwise None.
    """
    if categories is not None:
        df[column] = (
            df[column].astype("category").cat.set_categories(categories).cat.codes
        )
        return None

    df[column] = df[column].astype("category")
    categories_index = df[column].cat.categories
    df[column] = df[column].cat.codes
    return list(categories_index.tolist())


def broadcast_na(df: pd.DataFrame, indices: pd.DataFrame) -> pd.DataFrame:
    """Broadcasts rows containing NaN values by merging on non-NaN columns.

    Groups the input DataFrame by the presence of NaN values in columns matching
    the indices columns. For each group, it merges with the indices DataFrame
    using the non-NaN columns as keys.

    Args:
        df: The input DataFrame containing data and potentially NaN values.
        indices: The DataFrame defining the indices/keys to merge against.

    Returns:
        pd.DataFrame: A concatenated DataFrame representing the merged output.
    """
    processed = []
    na_patterns = pd.Series(
        list(df[indices.columns].isna().itertuples(index=False, name=None)),
        index=df.index,
    )
    for column_isna, df_subset in df.groupby(na_patterns):
        existing = [key for isna, key in zip(column_isna, indices.columns) if not isna]
        missing = [key for isna, key in zip(column_isna, indices.columns) if isna]
        processed.append(
            indices.merge(df_subset.drop(columns=missing), how="inner", on=existing)
        )
    return pd.concat(processed)


def resolve_path(
    path: str | os.PathLike, base_dir: str | os.PathLike | None = None
) -> Path:
    """Resolves a file or directory path.

    If the provided path is relative and a base directory is given, the path is
    resolved relative to the base directory. If the path is already absolute,
    it is returned unchanged as a Path object.

    Args:
        path: Path string or Path-like object.
        base_dir: Optional base directory to resolve relative paths against.

    Returns:
        Path: The resolved Path object.
    """
    p = Path(path)
    if not p.is_absolute() and base_dir is not None:
        return Path(base_dir) / p
    return p
