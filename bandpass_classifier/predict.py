"""Prediction entrypoint for the Bandpass Amplitude Anomaly Classifier.

This script runs the trained XGBoost model on a provided bandpass calibration table
to identify potential amplitude anomalies.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import tomllib
from xgboost import XGBClassifier

from .features.extractor import extract_paired_features, initialize_feature_extractor
from .io_utils import filter_degenerate_row, get_full_dataframe
from .utils import convert_to_category, get_env_vars_help_epilog, resolve_path


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
    parser.add_argument(
        "--output_prediction_path",
        type=Path,
        required=True,
        help="Path to output the prediction as a CSV format.",
    )
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)
    bandpass_table_df = get_full_dataframe(args.bandpass_table)
    bandpass_table_df = filter_degenerate_row(bandpass_table_df)

    initialize_feature_extractor(config)
    all_paired_features = extract_paired_features(bandpass_table_df, config)

    if all_paired_features.empty:
        print("No valid spectrum pairs found in the bandpass table to predict.")
        return

    categories_path = resolve_path(
        config["model"]["column_categories"], args.config.parent
    )
    with open(categories_path, "r") as f:
        column_categories = json.load(f)
    for column, categories in column_categories.items():
        convert_to_category(all_paired_features, column, categories)

    model = XGBClassifier()
    model_path = resolve_path(config["model"]["path"], args.config.parent)
    model.load_model(str(model_path))

    prediction = model.predict(all_paired_features)
    prediction = pd.Series(
        prediction, index=all_paired_features.index, dtype=bool, name="Prediction"
    )

    prediction.to_frame().to_csv(args.output_prediction_path)
    print(all_paired_features[prediction])


if __name__ == "__main__":
    main()
