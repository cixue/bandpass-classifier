"""Training script for the Bandpass Amplitude Anomaly Classifier.

This script loads raw bandpass tables, extracts features, maps labels,
partitions the dataset, prepares categorical features, and trains
an XGBoost classifier to predict anomalies in bandpass amplitude spectra.
"""

import argparse
import glob
import itertools
import json
import os
from pathlib import Path
import tomllib
from typing import cast, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import fbeta_score, confusion_matrix
from xgboost import XGBClassifier

from .features.extractor import extract_paired_features, initialize_feature_extractor
from .io_utils import get_flagtemplate_dataframe, get_full_dataframe
from .utils import broadcast_na, convert_to_category


def get_all_bandpass_table_df(config: dict) -> pd.DataFrame:
    """Concatenates all bandpass tables matching the pattern defined in configuration.

    Args:
        config: The model configuration dictionary.

    Returns:
        pd.DataFrame: A unified DataFrame containing all loaded bandpass data.
    """
    all_bandpass_table_path = list(
        map(Path, glob.glob(config["training"]["data"]["bandpass_table_pattern"]))
    )
    return pd.concat([get_full_dataframe(path) for path in all_bandpass_table_path])


def get_full_indices(
    full_indices: pd.MultiIndex, indices_df: pd.DataFrame
) -> pd.MultiIndex:
    """Reconstructs full multi-indices based on a sub-selected/aggregated level DataFrame.

    Filters the full index framework down to the rows matching the values in the
    aggregated level columns of indices_df.

    Args:
        full_indices: The original MultiIndex of the complete dataset.
        indices_df: The DataFrame specifying the target index values.

    Returns:
        pd.MultiIndex: The filtered MultiIndex containing only the matching rows.
    """
    aggregated_level = indices_df.columns.tolist()
    indices = indices_df.set_index(aggregated_level).index
    return cast(
        pd.MultiIndex,
        full_indices.to_frame(index=False)
        .set_index(aggregated_level)
        .loc[indices]
        .reset_index()
        .set_index(full_indices.names)
        .index
    )


def split_indices(
    paired_features: pd.DataFrame,
    label: pd.Series,
    config: dict,
) -> Tuple[pd.MultiIndex, pd.MultiIndex]:
    """Splits MultiIndex into training and test indices based on aggregated data level.

    Args:
        paired_features: The DataFrame containing paired feature representations.
        label: The target labels.
        config: The configuration dictionary.

    Returns:
        Tuple[pd.MultiIndex, pd.MultiIndex]: A tuple containing the training MultiIndex
            and the test MultiIndex.
    """
    label_aggregated_level = label.groupby(
        config["training"]["strategy"]["data_aggregation_level"]
    ).apply(any)
    (
        train_indices_aggregated_level,
        test_indices_aggregated_level,
    ) = train_test_split(
        label_aggregated_level.index.to_frame(index=False),
        test_size=config["training"]["strategy"]["test_size"],
        random_state=config["training"]["strategy"]["random_state"],
        stratify=label_aggregated_level,
    )
    train_indices = get_full_indices(
        cast(pd.MultiIndex, paired_features.index), train_indices_aggregated_level
    )
    test_indices = get_full_indices(
        cast(pd.MultiIndex, paired_features.index), test_indices_aggregated_level
    )
    return train_indices, test_indices


def extract_data(
    features: pd.DataFrame,
    label: pd.Series,
    indices: pd.MultiIndex,
    symmetric: bool,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extracts features and labels for given indices, optionally applying symmetry.

    Symmetry swaps corresponding paired columns (ending in '_0' and '_1') to augment
    the dataset with both permutation directions.

    Args:
        features: The source features DataFrame.
        label: The source label Series.
        indices: The MultiIndex targeting specific rows.
        symmetric: If True, duplicate data and permute the paired feature columns.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Extracted features and label series.
    """
    X = features.loc[indices]
    y = label.loc[indices]
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

    if symmetric:
        rename_dict = {}
        for col in features.columns:
            if col.endswith("_0"):
                col_0 = col
                col_1 = col.removesuffix("_0") + "_1"
                rename_dict[col_0] = col_1
                rename_dict[col_1] = col_0
        X = pd.concat([X, X.rename(columns=rename_dict)])
        y = pd.concat([y, y])

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    return X, y


def prepare_data(
    paired_features: pd.DataFrame,
    label: pd.Series,
    train_indices: pd.MultiIndex,
    test_indices: pd.MultiIndex,
    config: dict,
) -> Tuple[
    Dict[str, Optional[List[str]]],
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """Prepares and formats training and test datasets.

    Extracts training and testing splits, performs categorical column encoding,
    and returns mappings for the categories.

    Args:
        paired_features: The complete paired features DataFrame.
        label: The target labels Series.
        train_indices: MultiIndex for the training partition.
        test_indices: MultiIndex for the testing partition.
        config: The configuration dictionary.

    Returns:
        Tuple:
            - Dict[str, Optional[List[str]]]: Map of column names to category values.
            - pd.DataFrame: Training features.
            - pd.Series: Training labels.
            - pd.DataFrame: Testing features.
            - pd.Series: Testing labels.
    """
    X_train, y_train = extract_data(
        paired_features,
        label,
        train_indices,
        symmetric=config["training"]["strategy"]["symmetric"],
    )
    X_test, y_test = extract_data(
        paired_features,
        label,
        test_indices,
        symmetric=config["training"]["evaluation"]["symmetric"],
    )

    column_categories = dict()
    for cat_feature in config["features"]["categorical_features"]:
        if cat_feature in config["features"]["shared_features"]:
            cat_columns = [cat_feature]
        else:
            cat_columns = [cat_feature + "_0", cat_feature + "_1"]
        for cat_column in cat_columns:
            column_categories[cat_column] = convert_to_category(X_train, cat_column)
            convert_to_category(X_test, cat_column, column_categories[cat_column])

    return column_categories, X_train, y_train, X_test, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
    hyperparameter_updates: Optional[dict] = None,
) -> XGBClassifier:
    """Trains an XGBClassifier using configured parameters and optional updates.

    Calculates scale_pos_weight based on label ratio in training set to handle
    class imbalance.

    Args:
        X_train: Training features DataFrame.
        y_train: Training labels Series.
        X_val: Validation/Testing features DataFrame.
        y_val: Validation/Testing labels Series.
        config: Configuration dictionary.
        hyperparameter_updates: Optional dictionary of hyperparameters to override.

    Returns:
        XGBClassifier: The trained XGBoost classifier.
    """
    counts = y_train.value_counts()
    num_neg = counts.get(False, 0)
    num_pos = counts.get(True, 0)
    ratio = num_neg / num_pos if num_pos > 0 else 1.0

    weight_factor = config["training"]["strategy"].get("weight_factor", 1.0)
    scale_pos_weight = ratio * weight_factor

    model_params = config["training"]["hyperparameters"].copy()
    if hyperparameter_updates:
        model_params.update(hyperparameter_updates)

    model = XGBClassifier(
        tree_method="hist",  # Required for categorical support
        enable_categorical=True,  # Tells XGB to handle categories automatically
        scale_pos_weight=scale_pos_weight,  # Gives more weight to the minority class
        **model_params
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


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


def compute_f2_score(y_true, y_pred):
    """Computes the F2 score."""
    return fbeta_score(y_true, y_pred, beta=2, zero_division=0)


def get_group_stratified_folds(
    paired_features: pd.DataFrame,
    label: pd.Series,
    config: dict,
    n_splits: int,
    random_state: Optional[int] = None
):
    """Generates group-stratified splits on paired_features MultiIndex."""
    group_cols = config["training"]["strategy"]["data_aggregation_level"]
    label_aggregated_level = label.groupby(group_cols).apply(any)
    
    actual_splits = min(n_splits, len(label_aggregated_level))
    if actual_splits < 2:
        # Fallback: if we only have 0 or 1 groups, return a single fold using the whole set
        yield cast(pd.MultiIndex, paired_features.index), cast(pd.MultiIndex, paired_features.index)
        return
        
    try:
        skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
        splits = list(skf.split(label_aggregated_level.index.to_frame(index=False), label_aggregated_level))
    except Exception:
        kf = KFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
        splits = list(kf.split(label_aggregated_level.index.to_frame(index=False)))
        
    for train_group_idx, val_group_idx in splits:
        train_groups = label_aggregated_level.index[train_group_idx].to_frame(index=False)
        val_groups = label_aggregated_level.index[val_group_idx].to_frame(index=False)
        
        train_fold_indices = get_full_indices(cast(pd.MultiIndex, paired_features.index), train_groups)
        val_fold_indices = get_full_indices(cast(pd.MultiIndex, paired_features.index), val_groups)
        
        yield train_fold_indices, val_fold_indices



def tune_hyperparameters(
    paired_features: pd.DataFrame,
    label: pd.Series,
    train_indices: pd.MultiIndex,
    config: dict
) -> dict:
    """Finds the best hyperparameters for the given training indices using CV."""
    tuning_config = config["training"]["tuning"]
    method = tuning_config.get("method", "grid_search")
    n_splits = tuning_config.get("inner_folds", 5)
    random_state = config["training"]["strategy"].get("random_state")
    
    subset_features = paired_features.loc[train_indices]
    assert isinstance(subset_features, pd.DataFrame)
    subset_label = label.loc[train_indices]
    
    inner_folds_indices = list(get_group_stratified_folds(
        subset_features, subset_label, config, n_splits=n_splits, random_state=random_state
    ))
    
    if method == "grid_search":
        space = tuning_config.get("grid_search_space", {})
        keys = list(space.keys())
        values = [space[k] for k in keys]
        candidates = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        
        best_score = -1.0
        best_params = {}
        
        for params in candidates:
            scores = []
            for train_fold_idx, val_fold_idx in inner_folds_indices:
                _, X_tr, y_tr, X_val, y_val = prepare_data(
                    subset_features, subset_label, train_fold_idx, val_fold_idx, config
                )
                model = train_model(X_tr, y_tr, X_val, y_val, config, params)
                preds = model.predict(X_val)
                score = compute_f2_score(y_val, preds)
                scores.append(score)
                
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                
        print(f"GridSearch best inner score: {best_score:.4f} with params: {best_params}")
        return best_params

    elif method == "optuna":
        space = tuning_config.get("optuna_space", {})
        n_trials = space.get("n_trials", 10)
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial):
            params = {}
            for k, bounds in space.items():
                if k == "n_trials":
                    continue
                if isinstance(bounds, list) and len(bounds) == 2:
                    if isinstance(bounds[0], int) and isinstance(bounds[1], int):
                        params[k] = trial.suggest_int(k, bounds[0], bounds[1])
                    else:
                        params[k] = trial.suggest_float(k, bounds[0], bounds[1])
                else:
                    params[k] = trial.suggest_categorical(k, bounds)
                    
            scores = []
            for train_fold_idx, val_fold_idx in inner_folds_indices:
                _, X_tr, y_tr, X_val, y_val = prepare_data(
                    subset_features, subset_label, train_fold_idx, val_fold_idx, config
                )
                model = train_model(X_tr, y_tr, X_val, y_val, config, params)
                preds = model.predict(X_val)
                score = compute_f2_score(y_val, preds)
                scores.append(score)
                
            return np.mean(scores)
            
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        print(f"Optuna best inner score: {study.best_value:.4f} with params: {study.best_params}")
        return study.best_params

    else:
        raise ValueError(f"Unknown tuning method: {method}")


def run_nested_cv(
    paired_features: pd.DataFrame,
    label: pd.Series,
    config: dict,
    config_dir: Path
) -> List[float]:
    """Runs nested cross validation and returns outer fold test F2 scores."""
    tuning_config = config["training"]["tuning"]
    outer_folds = tuning_config.get("outer_folds", 5)
    random_state = config["training"]["strategy"].get("random_state")

    
    print(f"Running nested CV with {outer_folds} outer folds...")
    
    output_dir = config_dir / "intermediate_output"
    os.makedirs(output_dir, exist_ok=True)
    
    outer_folds_indices = list(get_group_stratified_folds(
        paired_features, label, config, n_splits=outer_folds, random_state=random_state
    ))
    
    outer_scores = []
    confusion_matrices = []
    results_list = []
    
    for i, (train_outer_idx, test_outer_idx) in enumerate(outer_folds_indices):
        print(f"--- Outer Fold {i+1}/{outer_folds} ---")
        
        best_params = tune_hyperparameters(
            paired_features, label, train_outer_idx, config
        )
        
        _, X_tr_outer, y_tr_outer, X_te_outer, y_te_outer = prepare_data(
            paired_features, label, train_outer_idx, test_outer_idx, config
        )
        model = train_model(X_tr_outer, y_tr_outer, X_te_outer, y_te_outer, config, best_params)
        
        preds_outer = model.predict(X_te_outer)
        score_outer = compute_f2_score(y_te_outer, preds_outer)
        print(f"Outer Fold {i+1} Test F2 Score: {score_outer:.4f}")
        outer_scores.append(score_outer)
        
        # Calculate confusion matrix for this fold
        cm = confusion_matrix(y_te_outer, preds_outer, labels=[False, True])
        confusion_matrices.append(cm)
        
        # Determine classification type (TP, FP, FN, TN) for the original test indices
        N = len(test_outer_idx)
        y_te_orig = y_te_outer.iloc[:N]
        preds_orig = preds_outer[:N]
        preds_series = pd.Series(preds_orig, index=y_te_orig.index, dtype=bool)
        
        fold_results = pd.DataFrame(index=y_te_orig.index)
        fold_results["fold"] = i + 1
        
        classification = pd.Series(index=y_te_orig.index, dtype=str)
        classification.loc[(y_te_orig == True) & (preds_series == True)] = "TP"
        classification.loc[(y_te_orig == False) & (preds_series == True)] = "FP"
        classification.loc[(y_te_orig == True) & (preds_series == False)] = "FN"
        classification.loc[(y_te_orig == False) & (preds_series == False)] = "TN"
        fold_results["classification"] = classification
        results_list.append(fold_results)
        
        # Save intermediate output for the outer loop fold in a subdirectory
        fold_dir = output_dir / f"fold_{i+1}"
        os.makedirs(fold_dir, exist_ok=True)
        
        hyp_path = fold_dir / "hyperparameters.json"
        with open(hyp_path, "w") as f:
            json.dump(best_params, f, indent=4)
            
        model_path = fold_dir / "model.json"
        model.save_model(str(model_path))
        
        indices_path = fold_dir / "test_indices.parquet"
        test_indices_df = test_outer_idx.to_frame(index=False)
        test_indices_df.to_parquet(indices_path)
        
    if results_list:
        all_results_df = pd.concat(results_list).reset_index()
        columns_to_keep = ["eb_uid", "spw_name_ms", "antenna_name", "fold", "classification"]
        all_results_df = all_results_df[columns_to_keep]
        csv_path = output_dir / "testing_results.csv"
        all_results_df.to_csv(csv_path, index=False)
        
    print(f"Nested CV completed. Outer fold F2 scores: {outer_scores}")
    print(f"Mean Outer F2 Score: {np.mean(outer_scores):.4f} +/- {np.std(outer_scores):.4f}")
    
    if confusion_matrices:
        mean_cm = np.mean(confusion_matrices, axis=0).tolist()
        std_cm = np.std(confusion_matrices, axis=0).tolist()
        sem_cm = (np.std(confusion_matrices, axis=0) / np.sqrt(len(confusion_matrices))).tolist()
        
        print("Average Confusion Matrix (Mean +/- Std):")
        print(f"  Negative (Actual) [TN, FP]: [{mean_cm[0][0]:.2f} +/- {std_cm[0][0]:.2f}, {mean_cm[0][1]:.2f} +/- {std_cm[0][1]:.2f}]")
        print(f"  Positive (Actual) [FN, TP]: [{mean_cm[1][0]:.2f} +/- {std_cm[1][0]:.2f}, {mean_cm[1][1]:.2f} +/- {std_cm[1][1]:.2f}]")
        
        cm_path = output_dir / "average_confusion_matrix.json"
        with open(cm_path, "w") as f:
            json.dump({
                "mean_confusion_matrix": mean_cm,
                "std_confusion_matrix": std_cm,
                "sem_confusion_matrix": sem_cm,
                "label_order": [False, True]
            }, f, indent=4)
            
    return outer_scores


def load_and_prepare_labels(
    bandpass_table_df: pd.DataFrame,
    paired_features: pd.DataFrame,
    config: dict
) -> pd.Series:
    """Loads flagtemplates, broadcasts anomalies, and maps label classifications.

    Args:
        bandpass_table_df: Source DataFrame of bandpass tables.
        paired_features: DataFrame containing paired feature representations.
        config: Configuration dictionary.

    Returns:
        pd.Series: A Series of boolean labels corresponding to the paired features index.
    """
    all_flagtemplate_path = list(
        map(Path, glob.glob(config["training"]["data"]["flagtemplate_pattern"]))
    )
    if not all_flagtemplate_path:
        raise ValueError("No flagtemplate files found matching pattern.")
    flagtemplate_df = pd.concat(map(get_flagtemplate_dataframe, all_flagtemplate_path))

    # Broadcast anomalies based on flagtemplates
    broadcasted_flagtemplate_df = broadcast_na(
        flagtemplate_df.reset_index(),
        bandpass_table_df.index.to_frame(index=False),
    )

    # Map reasons to True/False based on configuration
    anomaly_reasons = set(config["training"]["data"]["anomaly_reasons"])
    paired_level = list(paired_features.index.names)
    all_label = (
        broadcasted_flagtemplate_df["reason"]
        .isin(anomaly_reasons)
        .groupby([broadcasted_flagtemplate_df[col] for col in paired_level])
        .any()
        .reindex(paired_features.index, fill_value=False)
        .rename("label")
    )
    return all_label


def main() -> None:
    """Main training execution block.

    Parses CLI args, loads dataset, extracts features, performs splits, prepares data,
    trains the classifier, and saves output model/metadata artifacts.
    """
    parser = argparse.ArgumentParser("Bandpass Amplitude Anomaly Classifier")
    parser.add_argument(
        "--config", type=Path, required=True, help="TOML model config file."
    )
    parser.add_argument(
        "--sample",
        type=float,
        help="Sample fraction (if < 1.0) or count of rows (if >= 1.0) of input data.",
    )
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    bandpass_table_df = get_all_bandpass_table_df(config)

    if args.sample is not None:
        if args.sample < 1.0:
            bandpass_table_df = bandpass_table_df.iloc[:int(len(bandpass_table_df) * args.sample)]
        else:
            bandpass_table_df = bandpass_table_df.iloc[:int(args.sample)]

    assert isinstance(bandpass_table_df, pd.DataFrame)

    # Filter out degenerate amplitude rows and completely flagged rows
    initial_len = len(bandpass_table_df)
    bandpass_table_df = bandpass_table_df[
        ~bandpass_table_df["CPARAM"].apply(is_amplitude_degenerate)
    ]
    bandpass_table_df = bandpass_table_df[
        ~bandpass_table_df["flag_array"].apply(is_completely_flagged)
    ]
    print(f"Filtered out {initial_len - len(bandpass_table_df)} degenerate/flagged rows. Remaining: {len(bandpass_table_df)}")

    initialize_feature_extractor(config)

    all_paired_features = extract_paired_features(bandpass_table_df, config)

    all_label = load_and_prepare_labels(bandpass_table_df, all_paired_features, config)
    print("Label value counts:")
    print(all_label.value_counts())

    common_indices = (
        all_paired_features.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .index.intersection(all_label.index)
    )
    paired_features = all_paired_features.loc[common_indices]
    label = all_label.loc[common_indices]

    assert isinstance(paired_features, pd.DataFrame)
    assert isinstance(label, pd.Series)

    tuning_config = config["training"].get("tuning", {})
    tuning_enabled = tuning_config.get("enabled", False)

    if tuning_enabled:
        # 1. Run nested CV to compute generalized outer loop scores
        config_dir = args.config.parent
        run_nested_cv(paired_features, label, config, config_dir)
        
        # 2. Find best final hyperparameters on all data
        print("Tuning final model hyperparameters on all data...")
        best_params = tune_hyperparameters(paired_features, label, cast(pd.MultiIndex, paired_features.index), config)
        
        # 3. Train the final model using best found hyperparameters on all data
        column_categories, X_train, y_train, X_test, y_test = prepare_data(
            paired_features, label, cast(pd.MultiIndex, paired_features.index), cast(pd.MultiIndex, paired_features.index), config
        )
        model = train_model(X_train, y_train, X_test, y_test, config, best_params)
    else:
        column_categories, X_train, y_train, X_test, y_test = prepare_data(
            paired_features, label, cast(pd.MultiIndex, paired_features.index), cast(pd.MultiIndex, paired_features.index), config
        )
        model = train_model(X_train, y_train, X_test, y_test, config)

    model.save_model(config["model"]["path"])
    with open(config["model"]["column_categories"], "w") as f:
        json.dump(column_categories, f)


if __name__ == "__main__":
    main()

