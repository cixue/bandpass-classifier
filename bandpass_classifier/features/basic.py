"""Basic feature extraction functions for bandpass calibration spectra.

This module defines baseline features calculated directly from the spectrum,
including normalized median absolute deviation (NMAD) and spectral properties.
All extractor functions are registered using the `Extractor.register` decorator.
"""

import numpy as np
import pandas as pd

from bandpass_classifier.features.extractor import Extractor


def nmad(arr: np.ndarray) -> float:
    """Calculates the Normalized Median Absolute Deviation (NMAD).

    Args:
        arr: The input numpy array.

    Returns:
        float: The NMAD score of the array.
    """
    return 1.4826 * np.median(np.fabs(arr - np.median(arr)))


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


@Extractor.register(deps=["amplitude"])
def amp_nmad_diff4(df: pd.DataFrame) -> pd.Series:
    """Computes the NMAD of the difference between channels spaced 4 units apart.

    Args:
        df: A pandas DataFrame containing the 'amplitude' column.

    Returns:
        pd.Series: The NMAD score of the 4-channel differences.
    """
    return df["amplitude"].transform(lambda amp: nmad(amp[:-4] - amp[4:]))


@Extractor.register(deps=["amplitude"])
def amp_nmad(df: pd.DataFrame) -> pd.Series:
    """Computes the NMAD of the amplitude spectrum.

    Args:
        df: A pandas DataFrame containing the 'amplitude' column.

    Returns:
        pd.Series: The NMAD score of the amplitude.
    """
    return df["amplitude"].transform(nmad)


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

