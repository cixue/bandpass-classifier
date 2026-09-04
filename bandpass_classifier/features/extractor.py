"""Feature extraction engine for bandpass spectra.

This module provides the core `Extractor` class that manages feature registration,
calculates feature dependency graphs, and runs extraction pipelines in parallel.
It also provides initializer and wrapper functions to extract paired features.
"""

import math
import warnings
from collections.abc import Callable
from functools import partial
from graphlib import CycleError, TopologicalSorter
from typing import Any

import pandas as pd
from tqdm.contrib.concurrent import process_map

from ..utils import get_chunk_size, get_max_workers

__all__ = ["Extractor"]


class Extractor:
    """A registry and execution framework for feature extraction on DataFrames.

    Supports dependency resolution via topological sorting and parallel chunked processing.

    Attributes:
        _registry (Dict[str, Dict[str, Any]]): The internal registry mapping feature
            names to their specification (function, dependencies, external data requirements).
        _external_data_repository (Dict[str, Any]): Repository of global external data
            needed by extractor functions.
    """

    _registry: dict[str, dict[str, Any]] = {}
    _external_data_repository: dict[str, Any] = {}

    @classmethod
    def register(
        cls,
        name: str | None = None,
        deps: list[str] | None = None,
        external_data: list[str] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a feature extraction function.

        Args:
            name: Custom feature name. Defaults to the decorated function's name.
            deps: List of feature names this feature depends on.
            external_data: List of keys in the external data repository required by this feature.

        Returns:
            Callable: The decorator function wrapping the registered feature function.
        """
        if deps is None:
            deps = []

        if external_data is None:
            external_data = []

        def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
            feature_name = func.__name__ if name is None else name
            cls._registry[feature_name] = {
                "func": func,
                "deps": set(deps),
                "external_data": set(external_data),
            }
            return func

        return wrapper

    @classmethod
    def provide_external_data(cls, **provided_data: Any) -> None:
        """Injects external reference datasets into the extractor repository.

        Args:
            **provided_data: Arbitrary keyword arguments mapping data identifiers
                to their values.

        Raises:
            ValueError: If an external dataset key has already been registered.
        """
        for key, value in provided_data.items():
            if key in cls._external_data_repository:
                raise ValueError(f"External data {key} has already been provided.")
            cls._external_data_repository[key] = value

    @classmethod
    def reset_external_data(cls) -> None:
        """Clears the external data repository."""
        cls._external_data_repository.clear()

    @classmethod
    def extract(
        cls,
        df: pd.DataFrame,
        *,
        features: list[str],
        chunk_size: int | None = None,
        max_workers: int | None = None,
    ) -> pd.DataFrame:
        """Extracts specified features from the input DataFrame.

        Resolves dependencies topologically, splits the input DataFrame into chunks
        using round-robin interleaving, processes them in parallel, and gathers the requested features.

        Args:
            df: Input DataFrame.
            features: List of feature names to extract.
            chunk_size: Number of rows per parallel processing chunk. If None, checks
                the BANDPASS_CHUNK_SIZE environment variable before falling back to 8.
            max_workers: The maximum number of worker processes. If None, checks
                the BANDPASS_MAX_WORKERS environment variable before falling back
                to the default multiprocessing pool size.

        Returns:
            pd.DataFrame: A DataFrame containing the requested features in the original row order.

        Raises:
            ValueError: If dependencies or external data dependencies cannot be resolved.
            CycleError: If a cyclic dependency is detected among the features.
        """
        if max_workers is None:
            max_workers = get_max_workers()

        if chunk_size is None:
            chunk_size = get_chunk_size()

        # Make a copy to prevent modifying input
        requested_features = set(features)
        required_external_data: set[str] = set()

        required_features: set[str] = set()
        derived_features: set[str] = set()
        while requested_features:
            feature = requested_features.pop()
            if feature not in cls._registry:
                required_features.add(feature)
                continue
            derived_features.add(feature)
            requested_features.update(cls._registry[feature]["deps"] - derived_features)
            required_external_data.update(cls._registry[feature]["external_data"])

        missing_required_features = required_features - set(df.columns)
        if missing_required_features:
            raise ValueError(
                f"The following features cannot be derived and not provided: {missing_required_features}"
            )

        missing_required_external_data = required_external_data - set(
            cls._external_data_repository.keys()
        )
        if missing_required_external_data:
            raise ValueError(
                f"The following external data are not provided: {missing_required_external_data}"
            )

        try:
            extraction_order = list(
                TopologicalSorter(
                    {
                        feature: cls._registry[feature]["deps"] - required_features
                        for feature in derived_features
                    }
                ).static_order()
            )
        except CycleError as e:
            raise CycleError(f"Cyclic dependency found: {' -> '.join(e.args[1])}")

        if len(df) == 0:
            return pd.DataFrame(columns=features, index=df.index)

        if chunk_size <= 0:
            chunks = [df]
        else:
            num_chunks = math.ceil(len(df) / chunk_size)
            chunks = (
                [df.iloc[i::num_chunks] for i in range(num_chunks)]
                if num_chunks > 1
                else [df]
            )

        processed_chunks = process_map(
            partial(
                cls._process_chunk,
                required_features=required_features,
                extraction_order=extraction_order,
                registry=cls._registry,
                external_data_repository=cls._external_data_repository,
            ),
            chunks,
            chunksize=1,
            max_workers=max_workers,
        )

        # Strip out intermediate results and restore original row order.
        result = pd.concat(processed_chunks).loc[df.index][features]
        assert isinstance(result, pd.DataFrame)
        return result

    @staticmethod
    def _process_chunk(
        chunk: pd.DataFrame,
        *,
        required_features: set[str],
        extraction_order: list[str],
        registry: dict[str, dict[str, Any]],
        external_data_repository: dict[str, Any],
    ) -> pd.DataFrame:
        """Processes a single DataFrame chunk by calculating features in extraction order.

        Args:
            chunk: Subset DataFrame.
            required_features: Initial columns to copy from input chunk.
            extraction_order: Ordered list of features to evaluate.
            registry: The extractor registry dictionary.
            external_data_repository: Injectable external datasets.

        Returns:
            pd.DataFrame: Computed feature DataFrame.
        """
        # Extract all features including intermediate results.
        output_df = pd.DataFrame(index=chunk.index)
        for feature in required_features:
            output_df[feature] = chunk[feature]
        for feature in extraction_order:
            feature_spec = registry[feature]
            requested_columns = output_df[list(feature_spec["deps"])]
            requested_external_data = {
                key: external_data_repository[key]
                for key in feature_spec["external_data"]
            }
            output_df[feature] = feature_spec["func"](
                requested_columns, **requested_external_data
            )
        return output_df


def initialize_feature_extractor(config: dict) -> None:
    """Initializes the feature extractor with global configurations and external data.

    Args:
        config: The configuration dictionary.
    """
    Extractor.reset_external_data()
    Extractor.provide_external_data(**config["features"]["external_data"])


def extract_paired_features(
    bandpass_table_df: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """Extracts features and joins them into pairs based on configured settings.

    Args:
        bandpass_table_df: Raw input DataFrame.
        config: Configuration dictionary specifying features and pairings.

    Returns:
        pd.DataFrame: A DataFrame of paired features.
    """
    paired_level = config["features"]["paired_level"]
    if not bandpass_table_df.empty:
        bandpass_table_df = bandpass_table_df.groupby(paired_level).filter(
            lambda x: len(x) == 2
        )

    requested_features = (
        config["features"]["shared_features"] + config["features"]["spectrum_features"]
    )
    all_features = Extractor.extract(bandpass_table_df, features=requested_features)
    return pair_features(
        all_features,
        paired_level,
        config["features"]["shared_features"],
        config["features"]["spectrum_features"],
    )


def pair_features(
    features: pd.DataFrame,
    level: list[str],
    shared_features_list: list[str],
    spectrum_features_list: list[str],
) -> pd.DataFrame:
    """Groups features and combines spectrum pairs side-by-side.

    For spectrum features, columns are split and suffixed with '_0' and '_1' representing
    each spectrum in a pair. Shared features are group-deduplicated.

    Args:
        features: Computed features DataFrame.
        level: MultiIndex level names matching the spectrum pairs.
        shared_features_list: Columns representing identical properties in a pair.
        spectrum_features_list: Columns representing individual spectrum features.

    Returns:
        pd.DataFrame: Paired and joined features DataFrame.
    """
    expected_columns = shared_features_list + [
        f"{col}_{i}" for col in spectrum_features_list for i in (0, 1)
    ]
    if features.empty:
        return pd.DataFrame(
            columns=expected_columns,
            index=pd.MultiIndex.from_tuples([], names=level),
        )

    features = features.reset_index(
        [name for name in features.index.names if name not in level], drop=True
    )
    shared_features = (
        features[shared_features_list].groupby(features.index.names).first()
    )
    spectrum_features = features[spectrum_features_list]
    spectrum_features = (
        spectrum_features.groupby(level).filter(lambda x: len(x) == 2).copy()
    )
    if spectrum_features.empty:
        return pd.DataFrame(
            columns=expected_columns,
            index=pd.MultiIndex.from_tuples([], names=level),
        )

    spectrum_features["_row_number_"] = spectrum_features.groupby(
        spectrum_features.index.names
    ).cumcount()
    spectrum_features = spectrum_features.set_index(
        "_row_number_", append=True
    ).unstack("_row_number_")
    spectrum_features.columns = spectrum_features.columns.to_flat_index().map(
        lambda x: "_".join(map(str, x))
    )
    return shared_features.join(spectrum_features, how="inner")


def get_paired_features(
    features: pd.DataFrame,
    level: list[str],
    shared_features_list: list[str],
    spectrum_features_list: list[str],
) -> pd.DataFrame:
    """[Deprecated] Use `pair_features` instead."""
    warnings.warn(
        "`get_paired_features` is deprecated and will be removed in a future version. "
        "Use `pair_features` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return pair_features(
        features=features,
        level=level,
        shared_features_list=shared_features_list,
        spectrum_features_list=spectrum_features_list,
    )
