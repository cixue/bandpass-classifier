"""I/O utilities for reading and parsing bandpass data.

This module provides utility functions for reading and parsing CASA table formats,
extracting MeasurementSet/Execution Block UIDs, and parsing calibration
flagtemplate files.
"""

from pathlib import Path
import re
from typing import List, Literal, overload, Tuple, Union

# Note: casaconfig configuration must happen prior to importing casatools.table
# to properly suppress logs and auto-updates.
from casaconfig import config

config.logfile = None  # type: ignore # Suppress casa log file generation
config.data_auto_update = False  # type: ignore # Do not check for updates
config.measures_auto_update = False  # type: ignore

from casatools import table
import numpy as np
import pandas as pd

pd.options.future.infer_string = True


def get_eb_uid_from_filename(
    filename: str,
    matcher: re.Pattern = re.compile(r"uid___A002_X[0-9a-f]+_X[0-9a-f]+"),
) -> str:
    """Parses and formats the Execution Block (EB) UID from a given filename.

    Args:
        filename: The filename or path string containing the EB UID.
        matcher: A compiled regular expression pattern to search for the UID.

    Returns:
        str: The formatted EB UID (e.g., converting '___' to '://' and '_' to '/').

    Raises:
        ValueError: If the EB UID cannot be parsed from the filename.
    """
    result = matcher.search(filename)
    if result is None:
        raise ValueError(f"Cannot parse EB UID from filename: {filename}")
    return result.group().replace("___", "://").replace("_", "/")


@overload
def get_partial_dataframe(
    path: Union[Path, str],
    columns: List[str],
    include_eb_uid: Literal[False] = False,
) -> pd.DataFrame:
    ...


@overload
def get_partial_dataframe(
    path: Union[Path, str],
    columns: List[str],
    include_eb_uid: Literal[True],
) -> Tuple[str, pd.DataFrame]:
    ...


@overload
def get_partial_dataframe(
    path: Union[Path, str],
    columns: List[str],
    include_eb_uid: bool,
) -> Union[pd.DataFrame, Tuple[str, pd.DataFrame]]:
    ...


def get_partial_dataframe(
    path: Union[Path, str], columns: List[str], include_eb_uid: bool = False
) -> Union[pd.DataFrame, Tuple[str, pd.DataFrame]]:
    """Loads specified columns from a CASA table into a pandas DataFrame.

    Args:
        path: Path to the CASA table.
        columns: List of column names to load.
        include_eb_uid: If True, also extracts and returns the EB UID from the MSName keyword.

    Returns:
        Union[pd.DataFrame, Tuple[str, pd.DataFrame]]: The loaded DataFrame, or a tuple
            of (eb_uid, DataFrame) if include_eb_uid is True.
    """
    tb = table()
    tb.open(str(path))
    try:
        returned_df = pd.DataFrame.from_records(
            tb.getcoliter(columns, torecord=True)
        ).map(np.squeeze)
        if include_eb_uid:
            eb_uid = get_eb_uid_from_filename(tb.getkeyword("MSName"))
            return eb_uid, returned_df
        return returned_df
    finally:
        tb.close()


def get_full_dataframe(path: Union[Path, str]) -> pd.DataFrame:
    """Loads and compiles full bandpass calibration data from a table path.

    Merges bandpass data, spectral window configuration, and antenna details.

    Args:
        path: Path to the main bandpass table.

    Returns:
        pd.DataFrame: A pandas DataFrame indexed by ['eb_uid', 'spw_name_ms',
            'antenna_name', 'pol_id'].
    """
    if isinstance(path, str):
        path = Path(path)

    BANDPASS_COLUMNS = ["SPECTRAL_WINDOW_ID", "ANTENNA1", "CPARAM", "FLAG"]
    SPW_COLUMNS = ["NAME", "CHAN_FREQ"]
    ANTENNA_COLUMNS = ["NAME"]

    RETURNED_COLUMNS = {
        "SPECTRAL_WINDOW_ID": "spw_name_ms",
        "antenna_NAME": "antenna_name",
        "CPARAM": "CPARAM",
        "spw_CHAN_FREQ": "frequency_array",
        "FLAG": "flag_array",
        "spw_NAME": "spw_name",
    }

    eb_uid, bandpass_table = get_partial_dataframe(
        path, BANDPASS_COLUMNS, include_eb_uid=True
    )
    spw_table = get_partial_dataframe(path / "SPECTRAL_WINDOW", SPW_COLUMNS).rename(
        columns=lambda col: f"spw_{col}"
    )
    antenna_table = get_partial_dataframe(path / "ANTENNA", ANTENNA_COLUMNS).rename(
        columns=lambda col: f"antenna_{col}"
    )

    full_table = (
        bandpass_table.merge(
            spw_table,
            left_on="SPECTRAL_WINDOW_ID",
            right_index=True,
            validate="many_to_one",
        )
        .merge(
            antenna_table,
            left_on="ANTENNA1",
            right_index=True,
            validate="many_to_one",
        )
        .rename(columns=RETURNED_COLUMNS)
        .explode(["CPARAM", "flag_array"])
    )[RETURNED_COLUMNS.values()]

    full_table["eb_uid"] = eb_uid
    full_table["pol_id"] = full_table.groupby(level=0).cumcount()

    indices = ["eb_uid", "spw_name_ms", "antenna_name", "pol_id"]
    for index in indices:
        full_table[index] = full_table[index].astype(str)
    return full_table.set_index(indices)


def filter_degenerate_row(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out degenerate amplitude rows and completely flagged rows"""

    def is_amplitude_degenerate(cparam: np.ndarray) -> bool:
        """Checks if the amplitude (absolute value of CPARAM) is a single value, comparing to nanmean."""
        amp = np.absolute(cparam)
        if len(amp) == 0:
            return True
        mean_val = np.nanmean(amp)
        if np.isnan(mean_val):
            return True
        amp_filled = np.where(np.isnan(amp), mean_val, amp)
        return np.allclose(amp_filled, mean_val)

    def is_completely_flagged(flag_array: np.ndarray) -> bool:
        """Checks if the flag_array is entirely True."""
        if len(flag_array) == 0:
            return True
        return bool(np.all(flag_array))

    initial_len = len(df)
    df = df[~df["CPARAM"].apply(is_amplitude_degenerate)]
    df = df[~df["flag_array"].apply(is_completely_flagged)]
    print(
        f"Filtered out {initial_len - len(df)} degenerate/flagged rows. Remaining: {len(df)}"
    )
    return df


def get_flagtemplate_dataframe(path: Union[Path, str]) -> pd.DataFrame:
    """Parses a CASA flagtemplate file into a pandas DataFrame.

    Converts flag commands into structured rows and indexes them.

    Args:
        path: Path to the flagtemplate text file.

    Returns:
        pd.DataFrame: A pandas DataFrame indexed by ['eb_uid', 'spw_name_ms',
            'antenna_name', 'pol_id'].
    """
    if isinstance(path, str):
        path = Path(path)
    eb_uid = get_eb_uid_from_filename(path.name)
    rows = []
    with open(path, "r") as f:
        for line in f:
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith("#"):
                continue
            rows.append(
                {
                    key: value.strip("'")
                    for key, value in map(lambda kv: kv.split("="), line.split())
                }
            )
    returned_df = pd.DataFrame(rows).assign(eb_uid=eb_uid).fillna(pd.NA)

    for original_name, replaced_name in [
        ("antenna", "antenna_name"),
        ("spw", "spw_name_ms"),
        ("correlation", "pol_id"),
    ]:
        if original_name in returned_df.columns:
            returned_df.rename(columns={original_name: replaced_name}, inplace=True)

    for column in returned_df.columns:
        returned_df[column] = returned_df[column].apply(
            lambda e: e.split(",") if pd.notna(e) else e
        )
        returned_df = returned_df.explode(column)

    for column, mapping in [
        ("spw_name_ms", lambda x: x.split(":", maxsplit=1)[0]),
        ("pol_id", {"XX": "0", "YY": "1"}),
    ]:
        if column in returned_df.columns:
            returned_df[column] = returned_df[column].map(mapping, na_action="ignore")

    indices = ["eb_uid", "spw_name_ms", "antenna_name", "pol_id"]
    for index in indices:
        if index in returned_df.columns:
            returned_df[index] = returned_df[index].astype(str)
        else:
            returned_df = returned_df.assign(**{index: pd.NA})

    return returned_df.set_index(indices)
