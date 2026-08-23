"""Prediction entrypoint for the Bandpass Amplitude Anomaly Classifier.

This script runs the trained XGBoost model on a provided bandpass calibration table
to identify potential amplitude anomalies.
"""

import argparse
import json
from pathlib import Path
import tomllib

import pandas as pd
from xgboost import XGBClassifier

from .features.extractor import extract_paired_features, initialize_feature_extractor
from .io_utils import get_full_dataframe, filter_degenerate_row
from .utils import convert_to_category, get_env_vars_help_epilog


def main() -> None:
    """Parses command-line arguments and runs predictions on a bandpass table.

    Loads the configuration file, reads the bandpass table, initializes
    features, extracts paired features, maps categories, loads the model,
    and runs predictions.
    """
    parser = argparse.ArgumentParser(
        "Bandpass Amplitude Anomaly Classifier",
        epilog=get_env_vars_help_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="TOML model config file."
    )
    parser.add_argument(
        "--bandpass_table",
        type=Path,
        required=True,
        help="Bandpass table to be predicted.",
    )
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)
    bandpass_table_df = get_full_dataframe(args.bandpass_table)
    bandpass_table_df = filter_degenerate_row(bandpass_table_df)

    initialize_feature_extractor(config)
    all_paired_features = extract_paired_features(bandpass_table_df, config)

    with open(config["model"]["column_categories"], "r") as f:
        column_categories = json.load(f)
    for column, categories in column_categories.items():
        convert_to_category(all_paired_features, column, categories)

    model = XGBClassifier()
    model.load_model(config["model"]["path"])

    prediction = model.predict(all_paired_features)
    prediction = pd.Series(prediction, index=all_paired_features.index, dtype=bool)

    print(all_paired_features[prediction])


if __name__ == "__main__":
    main()

