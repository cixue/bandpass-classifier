"""Scan statistics features registered for the bandpass classifier.

This module maps raw and intermediate features of bandpass calibration solutions to
calculated scan statistics outputs (fixed, masked, and unmasked windows), providing
features such as window scores, starting and ending channel indices, and segment widths.
"""

from typing import Any, Dict

import pandas as pd

from bandpass_classifier.features.extractor import Extractor
from bandpass_classifier.features.scan_statistics import scan_statistics as ss
from bandpass_classifier.utils import memory


@memory().cache
def _cached_compute_scan_statistics_scores(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Helper wrapper to cache scan statistics computations.

    Args:
        *args: Positional arguments passed to compute_scan_statistics_scores.
        **kwargs: Keyword arguments passed to compute_scan_statistics_scores.

    Returns:
        Dict[str, Any]: Cached scan statistics results dictionary.
    """
    return ss.compute_scan_statistics_scores(*args, **kwargs)


@Extractor.register(
    deps=["frequency_array_GHz", "amplitude", "flag_array", "atmospheric_interference"]
)
def scan_statistics(df: pd.DataFrame) -> pd.Series:
    """Computes scan statistics over a DataFrame of calibration rows.

    Args:
        df: Input DataFrame containing columns 'amplitude', 'frequency_array_GHz',
            'flag_array', and 'atmospheric_interference'.

    Returns:
        pd.Series: Calculated scan statistics dictionaries for each row.
    """
    return df.apply(
        lambda row: _cached_compute_scan_statistics_scores(
            {
                "key": ss.Input(
                    amplitude=row["amplitude"],
                    frequency=row["frequency_array_GHz"],
                    flag_array=row["flag_array"],
                    atm_ranges=row["atmospheric_interference"],
                )
            }
        )["key"],
        axis=1,
    )


@Extractor.register(deps=["frequency_array_GHz"], external_data=["transmission_path"])
def atmospheric_interference(df: pd.DataFrame, *, transmission_path: str) -> pd.Series:
    """Detects frequency ranges corresponding to atmospheric absorption lines.

    Args:
        df: Input DataFrame containing 'frequency_array_GHz'.
        transmission_path: Path to the atmospheric transmission table.

    Returns:
        pd.Series: Lists of atmospheric interference channel index ranges.
    """
    trans_freqs, trans_vals = ss.load_transmission(transmission_path)
    return df["frequency_array_GHz"].transform(
        lambda freq_array: ss.detect_atm_ranges(freq_array, trans_freqs, trans_vals)
    )


# Top-level callable classes with __reduce__ to ensure they are picklable
class ScoreExtractor:
    """Extractor for the scan statistics score of a specific scan mode."""

    def __init__(self, scan_mode: str):
        self.scan_mode = scan_mode

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return df["scan_statistics"].apply(lambda x: x[self.scan_mode].score)

    def __reduce__(self):
        return (ScoreExtractor, (self.scan_mode,))


class WinStartExtractor:
    """Extractor for the scan statistics window start index of a specific scan mode."""

    def __init__(self, scan_mode: str):
        self.scan_mode = scan_mode

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return df["scan_statistics"].apply(lambda x: x[self.scan_mode].win_start)

    def __reduce__(self):
        return (WinStartExtractor, (self.scan_mode,))


class WinEndExtractor:
    """Extractor for the scan statistics window end index of a specific scan mode."""

    def __init__(self, scan_mode: str):
        self.scan_mode = scan_mode

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return df["scan_statistics"].apply(lambda x: x[self.scan_mode].win_end)

    def __reduce__(self):
        return (WinEndExtractor, (self.scan_mode,))


class SegmentWidthExtractor:
    """Extractor for physical segment width calculated from window boundaries."""

    def __init__(self, start_col: str, end_col: str):
        self.start_col = start_col
        self.end_col = end_col

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return df.apply(
            lambda row: (
                pd.NA
                if row[self.start_col] < 0
                or row[self.end_col] >= len(row["frequency_array"])
                else abs(
                    row["frequency_array"][row[self.start_col]]
                    - row["frequency_array"][row[self.end_col]]
                )
            ),
            axis="columns",
        )

    def __reduce__(self):
        return (SegmentWidthExtractor, (self.start_col, self.end_col))


# Loop to instantiate and register Fixed, Masked, and Unmasked features
for scan_mode in ["fixed", "masked", "unmasked"]:
    start_name = f"win_{scan_mode}_start"
    end_name = f"win_{scan_mode}_end"

    Extractor.register(name=f"score_{scan_mode}", deps=["scan_statistics"])(ScoreExtractor(scan_mode))
    Extractor.register(name=start_name, deps=["scan_statistics"])(WinStartExtractor(scan_mode))
    Extractor.register(name=end_name, deps=["scan_statistics"])(WinEndExtractor(scan_mode))
    Extractor.register(
        name=f"segment_width_{scan_mode}",
        deps=["frequency_array", start_name, end_name]
    )(SegmentWidthExtractor(start_name, end_name))