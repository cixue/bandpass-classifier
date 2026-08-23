"""Utility functions for the bandpass classifier.

This module provides common utilities for caching, category conversion, and missing value
broadcasting across pandas DataFrames.

Environment Variables:
    BANDPASS_ENABLE_CACHE: Set to '1', 'true', or 'yes' to enable joblib disk caching (default: disabled).
    BANDPASS_CACHE_DIR: Custom path for the joblib cache directory (default: '.cache').
"""

import functools
import os
from typing import Any, List, Optional

from joblib import Memory
import pandas as pd

_ENABLE_CACHE_ENV_VAR = "BANDPASS_ENABLE_CACHE"
_CACHE_DIR_ENV_VAR = "BANDPASS_CACHE_DIR"


def is_cache_enabled() -> bool:
    """Checks whether joblib caching is enabled via environment variable.

    Returns:
        bool: True if caching is enabled via BANDPASS_ENABLE_CACHE, False otherwise (default).
    """
    return os.environ.get(_ENABLE_CACHE_ENV_VAR, "").lower() in ("1", "true", "yes")


def get_cache_location() -> str:
    """Gets the directory location for joblib caching.

    Returns:
        str: The cache directory path specified by BANDPASS_CACHE_DIR, defaulting to '.cache'.
    """
    return os.environ.get(_CACHE_DIR_ENV_VAR, ".cache")


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
    df: pd.DataFrame, column: str, categories: Optional[List[str]] = None
) -> Optional[List[str]]:
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
        index=df.index
    )
    for column_isna, df_subset in df.groupby(na_patterns):
        existing = [key for isna, key in zip(column_isna, indices.columns) if not isna]
        missing = [key for isna, key in zip(column_isna, indices.columns) if isna]
        processed.append(
            indices.merge(df_subset.drop(columns=missing), how="inner", on=existing)
        )
    return pd.concat(processed)

