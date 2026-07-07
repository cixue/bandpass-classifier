"""Basic feature extraction functions for bandpass calibration spectra.

This module defines baseline features calculated directly from the spectrum,
including normalized median absolute deviation (NMAD) and spectral properties.
All extractor functions are registered using the `Extractor.register` decorator.
"""

import numpy as np
import pandas as pd

from bandpass_classifier.features.extractor import Extractor


def nmad(arr: np.ndarray) -> float:
    """Calculates the Normalized Median Absolute Deviation (NMAD) ignoring NaNs.

    Args:
        arr: The input numpy array.

    Returns:
        float: The NMAD score of the array.
    """
    return 1.4826 * np.nanmedian(np.fabs(arr - np.nanmedian(arr)))


@Extractor.register(deps=["spw_name"])
def receiver_band(df: pd.DataFrame) -> pd.Series:
    """Extracts the receiver band identifier from spectral window name.

    Args:
        df: A pandas DataFrame containing the 'spw_name' column.

    Returns:
        pd.Series: The extracted receiver band names.
    """
    return df["spw_name"].transform(lambda x: x.split("#")[1])


@Extractor.register(deps=["frequency_array"])
def frequency_array_GHz(df: pd.DataFrame) -> pd.Series:
    """Converts the frequency array from Hz to GHz.

    Args:
        df: A pandas DataFrame containing the 'frequency_array' column.

    Returns:
        pd.Series: The frequency arrays in GHz.
    """
    return df["frequency_array"] / 1e9


@Extractor.register(deps=["CPARAM"])
def amplitude(df: pd.DataFrame) -> pd.Series:
    """Calculates the absolute amplitude of the complex CPARAM parameter.

    Args:
        df: A pandas DataFrame containing the complex 'CPARAM' column.

    Returns:
        pd.Series: The absolute amplitude values.
    """
    return df["CPARAM"].transform(np.absolute)


@Extractor.register(deps=["amplitude", "flag_array"])
def amplitude_flagged(df: pd.DataFrame) -> pd.Series:
    """Flags amplitude with flagged values replaced with NaNs.

    Args:
        df: A pandas DataFrame containing the unflagged 'amplitude' and 'flag_array' columns.

    Returns:
        pd.Series: The flagged amplitude values.
    """
    return df.apply(
        lambda row: np.where(row["flag_array"], np.nan, row["amplitude"]), axis=1
    )


@Extractor.register(deps=["amplitude_flagged"])
def amp_nmad_diff4(df: pd.DataFrame) -> pd.Series:
    """Computes the NMAD of the difference between channels spaced 4 units apart.
    Channels containing NaN values do not contribute toward the calculation of `nmad_diff4`.

    Args:
        df: A pandas DataFrame containing the 'amplitude_flagged' column.

    Returns:
        pd.Series: The NMAD score of the 4-channel differences.
    """
    return df["amplitude_flagged"].transform(lambda amp: nmad(amp[:-4] - amp[4:]))


@Extractor.register(deps=["amplitude_flagged"])
def amp_nmad(df: pd.DataFrame) -> pd.Series:
    """Computes the NMAD of the amplitude_flagged spectrum.
    Channels containing NaN values do not contribute toward the calculation of `nmad`.

    Args:
        df: A pandas DataFrame containing the 'amplitude_flagged' column.

    Returns:
        pd.Series: The NMAD score of the amplitude_flagged.
    """
    return df["amplitude_flagged"].transform(nmad)


@Extractor.register(deps=["amp_nmad_diff4", "amp_nmad"])
def amp_norm_nmad_diff4(df: pd.DataFrame) -> pd.Series:
    """Normalizes the difference-4 NMAD feature by the overall amplitude NMAD.

    Args:
        df: A pandas DataFrame containing 'amp_nmad_diff4' and 'amp_nmad'.

    Returns:
        pd.Series: The normalized difference NMAD values.
    """
    return df["amp_nmad_diff4"] / df["amp_nmad"]


@Extractor.register(deps=["frequency_array"])
def channel_spacing(df: pd.DataFrame) -> pd.Series:
    """Calculates the median channel frequency spacing.

    Args:
        df: A pandas DataFrame containing the 'frequency_array' column.

    Returns:
        pd.Series: The median channel spacing.
    """
    return df["frequency_array"].transform(lambda arr: abs(np.median(np.diff(arr))))


@Extractor.register(deps=["frequency_array"])
def central_frequency(df: pd.DataFrame) -> pd.Series:
    """Calculates the central frequency of the spectrum.

    Args:
        df: A pandas DataFrame containing the 'frequency_array' column.

    Returns:
        pd.Series: The median frequency of the spectrum.
    """
    return df["frequency_array"].transform(lambda arr: np.median(arr))


@Extractor.register(deps=["amplitude_flagged"])
def amp_top_norm_discontinuity(df: pd.DataFrame) -> pd.Series:
    """Calculates the largest edge discontinuity in a spectrum. A edge discontinuity is
    defined to be the absolute lag-1 difference, with the search restricted to the first
    and last 10% of channels (bounded by the first/last valid, non-NaN indices),
    normalized by the MAD of the lag-1 differences computed over the entire spectrum.

    Args:
        df: A pandas DataFrame containing the 'amplitude_flagged' column.

    Returns:
        pd.Series: The top normalized edge discontinuity in amplitude_flagged.
    """

    def diff(arr, k=1):
        return arr[:-k] - arr[k:]

    def top_norm_discontinuity(arr):
        valid_indices = np.where(~np.isnan(arr))[0]
        min_index, max_index = valid_indices[0], valid_indices[-1]
        num_channel = max(4, int((max_index - min_index) * 0.1))
        arr_head = np.fabs(diff(arr[min_index : min_index + num_channel]))
        arr_tail = np.fabs(diff(arr[max_index + 1 - num_channel : max_index + 1]))
        arr_variation = nmad(diff(arr))
        return np.nanmax([arr_head, arr_tail]) / arr_variation

    return df["amplitude_flagged"].apply(top_norm_discontinuity)
