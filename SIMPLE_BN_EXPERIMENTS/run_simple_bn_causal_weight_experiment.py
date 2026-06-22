# AUTOMATED SEED x CAUSAL-WEIGHT EXPERIMENT
# SIMPLE-BN DATASET VERSION

import os
import random
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier
import dice_ml


import sys
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\irene\OneDrive\Υπολογιστής\TuE\BEP\BEP_CODE")
sys.path.insert(0, str(PROJECT_DIR))

from counterfactual_algorithms_dice.CAUSAL_DICE import DiceGeneticCausal
from counterfactual_algorithms_dice.DICE import DiceGenetic



warnings.filterwarnings("ignore")
DATA_PATH = None # this script generates the dataset
# 1. SETTINGS

OUTPUT_DIR = r"path\to\your\output\directory\results_will_be_saved_here"

target_col = "y"

SEEDS = list(range(10))


#CAUSAL_WEIGHTS = [0.05,0.1, 0.5, 0.7,0.85, 1.0, 1.5]
CAUSAL_WEIGHTS = [0.05, 0.1, 0.5, 0.7, 0.85, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0]

N_ROWS = 1000
N_INSTANCES = 100
MAXITERATIONS = 1000
STABILITY_K = 1000
STABILITY_SIGMA = 0.05



# 2. SIMPLE-BN FEATURE / SCM SETUP

continuous_features = ["x1", "x2", "x3", "x4", "x5"]
categorical_features = []

x1_col = "x1"
x2_col = "x2"
x3_col = "x3"
x4_col = "x4"
x5_col = "x5"

# x1 and x2 are causal parents of x3
# x4 and x5 are extra predictive variables for y
exogenous_features = [x1_col, x2_col]
endogenous_features = [x3_col]




def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def true_simple_bn_x3(row):
    """
    True deterministic structural part:
        x3 = 10 * ((x1 + x2)^2 / 180^2) + 10

    The generated dataset adds noise to x3, but the SCM penalty uses
    the deterministic structural equation.
    """
    return 10 * ((row["x1"] + row["x2"]) ** 2 / 180 ** 2) + 10



# 3. REPRODUCIBILITY HELPERS

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)



# 4. DATA / MODEL / SCM HELPERS


def generate_simple_bn_data(n_rows=1000, seed=42):
    """
    Generates an extended Simple-BN dataset:
        x1 ~ N(50, 15)
        x2 ~ N(50, 17)

        x3 = 10 * ((x1 + x2)^2 / 180^2) + 10 + noise

        x4 and x5 are extra variables that are correlated with y.
        They do not belong to the SCM relation x1, x2 -> x3.
    """
    rng = np.random.default_rng(seed)

    x1 = rng.normal(50, 15, n_rows)
    x2 = rng.normal(50, 17, n_rows)

    # Endogenous variable from the SCM
    x3 = 10 * ((x1 + x2) ** 2 / 180 ** 2) + 10 + rng.normal(0, 0.5, n_rows)

    # Extra variables correlated with the decision boundary
    x4 = 0.5 * x1 + 0.3 * x2 + rng.normal(0, 5, n_rows)
    x5 = 0.4 * x1 - 0.2 * x3 + rng.normal(0, 4, n_rows)

    # y depends on x1, x2, x3, x4, and x5
    y_score = (
        10.5 * ((x1 * x2) / 8100)
        + 10
        - x3
        + 0.04 * x4
        + 0.03 * x5
    )

    y_prob = sigmoid(y_score)
    y = (y_prob > 0.5).astype(int)

    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x4": x4,
        "x5": x5,
        "y": y
    })

    return df


def load_data(data_path=DATA_PATH, seed=42):
    if data_path is None:
        df = generate_simple_bn_data(n_rows=N_ROWS, seed=seed)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df[target_col] = df[target_col].astype(int)

    print("\nDataset shape:")
    print(df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nTarget distribution:")
    print(df[target_col].value_counts())
    print(df[target_col].value_counts(normalize=True))

    return df


def split_data(df, seed):
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy().astype(int)

    original_dtypes = X.dtypes.to_dict()

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y
    )

    X_train_initial, X_additional, y_train_initial, y_additional = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=seed,
        stratify=y_temp
    )

    X_train_retrained = pd.concat([X_train_initial, X_additional], axis=0)
    y_train_retrained = pd.concat([y_train_initial, y_additional], axis=0)

    print("\nSplit sizes:")
    print(f"Initial train set: {X_train_initial.shape[0]} rows")
    print(f"Additional set:    {X_additional.shape[0]} rows")
    print(f"Retrained set:     {X_train_retrained.shape[0]} rows")
    print(f"Test set:          {X_test.shape[0]} rows")

    return {
        "X": X,
        "y": y,
        "X_train_initial": X_train_initial,
        "X_additional": X_additional,
        "X_train_retrained": X_train_retrained,
        "X_test": X_test,
        "y_train_initial": y_train_initial,
        "y_additional": y_additional,
        "y_train_retrained": y_train_retrained,
        "y_test": y_test,
        "original_dtypes": original_dtypes
    }


def fit_scm_models(X_train, model_name="SCM"):
    """
    For Simple-BN, the true SCM is known. We do not fit a regression model.
    The R² reported here measures how well the deterministic SCM equation
    explains the noisy generated x3.
    """
    df_scm = X_train[["x1", "x2", "x3"]].dropna().copy()

    x3_expected = df_scm.apply(true_simple_bn_x3, axis=1)
    x3_r2 = r2_score(df_scm["x3"], x3_expected)
    residuals = df_scm["x3"] - x3_expected

    print(f"\n{model_name}")
    print("SCM relation: x1, x2 -> x3")
    print("x3 = 10 * ((x1 + x2)^2 / 180^2) + 10")
    print(f"SCM R² using deterministic equation: {x3_r2:.4f}")
    print(f"Residual mean: {residuals.mean():.4f}")
    print(f"Residual std:  {residuals.std():.4f}")

    scm = {
        "x3": {
            "parents": ["x1", "x2"],
            "func": lambda row: true_simple_bn_x3(row)
        }
    }

    scm_summary = {
        "scm_model_name": model_name,
        "scm_relation": "x1, x2 -> x3",
        "scm_r2_true_equation_to_x3": x3_r2,
        "scm_residual_mean": residuals.mean(),
        "scm_residual_std": residuals.std(),
    }

    return scm, None, scm_summary


def build_xgb_pipeline(categorical_features, continuous_features, seed):
    transformers = []

    if len(continuous_features) > 0:
        transformers.append(("num", "passthrough", continuous_features))

    if len(categorical_features) > 0:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", xgb)
    ])

    return model


def evaluate_model(model, X_eval, y_eval, model_name="Model", print_report=False):
    y_pred = model.predict(X_eval)

    positive_class_index = list(model.classes_).index(1)
    y_prob = model.predict_proba(X_eval)[:, positive_class_index]

    results = {
        "model": model_name,
        "accuracy": accuracy_score(y_eval, y_pred),
        "f1": f1_score(y_eval, y_pred, zero_division=0),
        "precision": precision_score(y_eval, y_pred, zero_division=0),
        "recall": recall_score(y_eval, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_eval, y_prob)
    }

    print(f"\n{model_name}")
    print(f"Accuracy : {results['accuracy']:.4f}")
    print(f"F1-score : {results['f1']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall   : {results['recall']:.4f}")
    print(f"ROC-AUC  : {results['roc_auc']:.4f}")

    if print_report:
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_eval, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_eval, y_pred, digits=4, zero_division=0))

    return results, y_pred, y_prob


def train_models(split, seed):
    model_1 = build_xgb_pipeline(categorical_features, continuous_features, seed)
    model_1.fit(split["X_train_initial"], split["y_train_initial"])

    results_1, y_pred_1, y_prob_1 = evaluate_model(
        model_1,
        split["X_test"],
        split["y_test"],
        model_name="Model 1 trained on 60%",
        print_report=False
    )

    model_2 = build_xgb_pipeline(categorical_features, continuous_features, seed)
    model_2.fit(split["X_train_retrained"], split["y_train_retrained"])

    results_2, y_pred_2, y_prob_2 = evaluate_model(
        model_2,
        split["X_test"],
        split["y_test"],
        model_name="Model 2 trained on 80%",
        print_report=False
    )

    model_metrics = pd.DataFrame([results_1, results_2])
    return model_1, model_2, model_metrics


def make_feature_ranges(X_train):
    return {
        col: (X_train[col].min(), X_train[col].max())
        for col in continuous_features
    }


def make_dice_objects(X_train, y_train, model):
    df_dice = pd.concat([X_train, y_train], axis=1)
    df_dice[target_col] = df_dice[target_col].astype(int)

    data_dice = dice_ml.Data(
        dataframe=df_dice,
        continuous_features=continuous_features,
        outcome_name=target_col
    )

    data_dice.continuous_feature_indexes = [
        data_dice.feature_names.index(f)
        for f in data_dice.continuous_feature_names
    ]

    model_dice = dice_ml.Model(
        model=model,
        backend="sklearn"
    )

    return data_dice, model_dice


# 5. COUNTERFACTUAL HELPERS

def fix_cf_types(cf_row, original_dtypes, continuous_features, categorical_features):
    cf_row = cf_row.copy()

    for col in continuous_features:
        if col in cf_row.index:
            cf_row[col] = float(cf_row[col])

    for col in categorical_features:
        if col in cf_row.index:
            dtype = original_dtypes[col]

            if pd.api.types.is_integer_dtype(dtype):
                cf_row[col] = int(round(float(cf_row[col])))
            elif pd.api.types.is_float_dtype(dtype):
                cf_row[col] = float(cf_row[col])
            else:
                cf_row[col] = str(cf_row[col])

    return cf_row


def get_first_cf(cf_object, target_col):
    cf_df = None

    if hasattr(cf_object, "final_cfs_df"):
        cf_df = cf_object.final_cfs_df

    elif hasattr(cf_object, "cf_examples_list"):
        if len(cf_object.cf_examples_list) == 0:
            return None

        first_example = cf_object.cf_examples_list[0]

        if hasattr(first_example, "final_cfs_df"):
            cf_df = first_example.final_cfs_df
        else:
            return None

    else:
        return None

    if cf_df is None or len(cf_df) == 0:
        return None

    cf_row = cf_df.iloc[0].copy()
    cf_row = cf_row.drop(labels=[target_col], errors="ignore")

    return cf_row


def counterfactual_stability(
    cf_row,
    model,
    continuous_features,
    feature_ranges,
    target_col=None,
    desired_class=1,
    K=1000,
    sigma=0.05,
    random_state=42,
    scm=None,
    endogenous_features=None
):
    """
    Simple-BN causal-neighbourhood stability:
    - perturb x1 and x2
    - recompute x3 from the SCM
    - then evaluate model probability around the CF

    """
    rng = np.random.default_rng(random_state)

    if isinstance(cf_row, pd.DataFrame):
        cf_row = cf_row.iloc[0]

    cf_row = cf_row.copy().astype(object)

    if target_col is not None:
        cf_row = cf_row.drop(labels=[target_col], errors="ignore")

    if endogenous_features is None:
        endogenous_features = []

    perturb_features = [
        col for col in continuous_features
        if col not in endogenous_features
    ]

    neighbourhood = []

    for _ in range(K):
        perturbed = cf_row.copy().astype(object)

        for col in perturb_features:
            low, high = feature_ranges[col]
            feature_range = high - low

            if feature_range == 0:
                continue

            noise = rng.normal(loc=0.0, scale=sigma * feature_range)
            new_value = float(cf_row[col]) + noise
            perturbed[col] = float(np.clip(new_value, low, high))

        if scm is not None:
            for endog_col in endogenous_features:
                if endog_col in scm:
                    expected_value = scm[endog_col]["func"](perturbed)
                    low, high = feature_ranges[endog_col]
                    perturbed[endog_col] = float(np.clip(expected_value, low, high))

        neighbourhood.append(perturbed)

    neighbourhood_df = pd.DataFrame(neighbourhood)
    neighbourhood_df = neighbourhood_df[cf_row.index]
    neighbourhood_df = neighbourhood_df.reset_index(drop=True)

    desired_class_index = list(model.classes_).index(desired_class)

    probs = model.predict_proba(neighbourhood_df)[:, desired_class_index]
    preds = model.predict(neighbourhood_df)

    mean_prob = np.mean(probs)
    std_prob = np.std(probs)

    return {
        "stability_score": mean_prob - std_prob,
        "mean_neighbourhood_prob": mean_prob,
        "std_neighbourhood_prob": std_prob,
        "min_neighbourhood_prob": np.min(probs),
        "max_neighbourhood_prob": np.max(probs),
        "validity_in_neighbourhood": np.mean(preds == desired_class)
    }


def describe_cf_changes(
    original_row,
    cf_row,
    continuous_features,
    categorical_features,
    feature_ranges=None,
    scm=None,
    endogenous_features=None,
    tol=1e-6
):
    changed_features = []
    numeric_change_sum = 0.0
    normalized_numeric_change_sum = 0.0
    categorical_change_count = 0
    feature_changes = {}

    if endogenous_features is None:
        endogenous_features = []

    for col in original_row.index:
        if col not in cf_row.index:
            continue

        old_val = original_row[col]
        new_val = cf_row[col]

        if col in continuous_features:
            diff = abs(float(new_val) - float(old_val))
            feature_changes[f"change_{col}"] = diff

            if feature_ranges is not None and col in feature_ranges:
                low, high = feature_ranges[col]
                feature_range = max(high - low, 1e-6)
                norm_diff = diff / feature_range
                feature_changes[f"normalized_change_{col}"] = norm_diff
            else:
                norm_diff = np.nan

            if diff > tol:
                changed_features.append(col)
                numeric_change_sum += diff

                if not np.isnan(norm_diff):
                    normalized_numeric_change_sum += norm_diff

        elif col in categorical_features:
            changed = int(old_val != new_val)
            feature_changes[f"change_{col}"] = changed

            if changed:
                changed_features.append(col)
                categorical_change_count += 1

    scm_violation_sum = 0.0
    normalized_scm_violation_sum = 0.0

    if scm is not None:
        for endog_col in endogenous_features:
            if endog_col not in scm or endog_col not in cf_row.index:
                continue

            expected_value = scm[endog_col]["func"](cf_row)
            actual_value = float(cf_row[endog_col])
            violation = abs(actual_value - expected_value)

            feature_changes[f"scm_expected_{endog_col}"] = expected_value
            feature_changes[f"scm_actual_{endog_col}"] = actual_value
            feature_changes[f"scm_violation_{endog_col}"] = violation

            scm_violation_sum += violation

            if feature_ranges is not None and endog_col in feature_ranges:
                low, high = feature_ranges[endog_col]
                feature_range = max(high - low, 1e-6)
                normalized_violation = violation / feature_range
            else:
                normalized_violation = np.nan

            feature_changes[f"normalized_scm_violation_{endog_col}"] = normalized_violation

            if not np.isnan(normalized_violation):
                normalized_scm_violation_sum += normalized_violation

    return {
        "n_changed_features": len(changed_features),
        "numeric_change_sum": numeric_change_sum,
        "normalized_numeric_change_sum": normalized_numeric_change_sum,
        "categorical_change_count": categorical_change_count,
        "changed_features": changed_features,
        "scm_violation_sum": scm_violation_sum,
        "normalized_scm_violation_sum": normalized_scm_violation_sum,
        **feature_changes
    }



# 6. RUN NORMAL OR CAUSAL COUNTERFACTUALS


def run_cf_stability_for_model(
    model,
    explainer,
    X_test,
    continuous_features,
    categorical_features,
    feature_ranges,
    original_dtypes,
    target_col,
    model_name,
    method_name,
    seed,
    causal_weight=np.nan,
    n_instances=100,
    maxiterations=1000,
    use_private_generate=True,
    stability_K=1000,
    stability_sigma=0.05,
    scm=None,
    endogenous_features=None
):
    results_all = []
    cf_rows = {}

    n_to_run = min(n_instances, len(X_test))

    n_no_cf_found = 0
    n_invalid_cf = 0
    n_exceptions = 0

    if endogenous_features is None:
        endogenous_features = []

    print(f"\n\nRUNNING {model_name} | seed={seed} | method={method_name} | causal_weight={causal_weight}")
    print(f"Generating CFs for {n_to_run} test instances...")

    for i in range(n_to_run):
        set_all_seeds(seed * 100000 + i)

        query_instance = X_test.iloc[[i]].copy()
        original_row = query_instance.iloc[0].copy()

        orig_pred = model.predict(query_instance)[0]
        desired_class = 1 - int(orig_pred)
        desired_class_index = list(model.classes_).index(desired_class)

        try:
            generation_kwargs = dict(
                total_CFs=1,
                desired_class="opposite",
                features_to_vary="all",
                permitted_range=None,
                initialization="kdtree",
                proximity_weight=0.2,
                sparsity_weight=0.2,
                diversity_weight=5.0,
                categorical_penalty=0.1,
                algorithm="DiverseCF",
                yloss_type="hinge_loss",
                diversity_loss_type="dpp_style:inverse_dist",
                feature_weights="inverse_mad",
                stopping_threshold=0.5,
                posthoc_sparsity_param=0.1,
                posthoc_sparsity_algorithm="binary",
                maxiterations=maxiterations,
                thresh=1e-2,
                verbose=False
            )

            if use_private_generate:
                cf_object = explainer._generate_counterfactuals(
                    query_instance=query_instance,
                    **generation_kwargs
                )
            else:
                cf_object = explainer.generate_counterfactuals(
                    query_instance,
                    **generation_kwargs
                )

            cf_row = get_first_cf(cf_object, target_col)

            if cf_row is None:
                n_no_cf_found += 1
                print(f"{model_name} - Instance {i}: no CF found")
                continue

            cf_row = cf_row.reindex(X_test.columns)

            if cf_row.isnull().any():
                n_no_cf_found += 1
                print(f"{model_name} - Instance {i}: CF has missing columns")
                continue

            cf_row = fix_cf_types(
                cf_row=cf_row,
                original_dtypes=original_dtypes,
                continuous_features=continuous_features,
                categorical_features=categorical_features
            )

            cf_df_for_pred = cf_row.to_frame().T

            cf_pred = model.predict(cf_df_for_pred)[0]
            cf_prob_desired = model.predict_proba(cf_df_for_pred)[0, desired_class_index]

            if int(cf_pred) != int(desired_class):
                n_invalid_cf += 1
                print(
                    f"{model_name} - Instance {i}: invalid CF "
                    f"(orig={orig_pred}, desired={desired_class}, cf_pred={cf_pred}, "
                    f"desired_prob={cf_prob_desired:.4f})"
                )
                continue

            stability_results = counterfactual_stability(
                cf_row=cf_row,
                model=model,
                continuous_features=continuous_features,
                feature_ranges=feature_ranges,
                desired_class=desired_class,
                K=stability_K,
                sigma=stability_sigma,
                random_state=seed * 100000 + i,
                scm=scm,
                endogenous_features=endogenous_features
            )

            change_results = describe_cf_changes(
                original_row=original_row,
                cf_row=cf_row,
                continuous_features=continuous_features,
                categorical_features=categorical_features,
                feature_ranges=feature_ranges,
                scm=scm,
                endogenous_features=endogenous_features
            )

            results_all.append({
                "seed": seed,
                "causal_weight": causal_weight,
                "instance": i,
                "method": method_name,
                "model_name": model_name,
                "orig_pred": orig_pred,
                "desired_class": desired_class,
                "cf_pred": cf_pred,
                "cf_prob_desired": cf_prob_desired,
                **stability_results,
                **change_results
            })

            cf_rows[i] = cf_row

            print(
                f"{model_name} - Instance {i}: valid CF found "
                f"(orig={orig_pred}, cf={cf_pred}, desired_prob={cf_prob_desired:.4f})"
            )

        except Exception as e:
            n_exceptions += 1
            print(f"{model_name} - Instance {i} failed: {e}")

    results_df = pd.DataFrame(results_all)

    n_valid_cfs = len(results_df)
    success_rate = n_valid_cfs / n_to_run if n_to_run > 0 else 0

    generation_summary = {
        "seed": seed,
        "causal_weight": causal_weight,
        "model": model_name,
        "method": method_name,
        "n_attempted": n_to_run,
        "n_valid_cfs": n_valid_cfs,
        "n_no_cf_found": n_no_cf_found,
        "n_invalid_cfs": n_invalid_cf,
        "n_exceptions": n_exceptions,
        "success_rate": success_rate
    }

    # print(f"\n GENERATION SUMMARY ({model_name}) ")
    # print(generation_summary)

    if n_valid_cfs > 0:
        metric_cols = [
            "stability_score",
            "mean_neighbourhood_prob",
            "std_neighbourhood_prob",
            "validity_in_neighbourhood",
            "n_changed_features",
            "numeric_change_sum",
            "normalized_numeric_change_sum",
            "scm_violation_sum",
            "normalized_scm_violation_sum"
        ]
        metric_cols = [c for c in metric_cols if c in results_df.columns]
    #     print(f"\n AVERAGE METRICS ({model_name}) ")
    #     print(results_df[metric_cols].mean(numeric_only=True))
    # else:
    #     print("No valid CFs found.")

    return results_df, cf_rows, generation_summary



# 7. SCM RESIDUAL CHECK FOR GENERATED COUNTERFACTUALS


def evaluate_scm_residuals_for_cfs(
    cf_dict,
    scm,
    endogenous_features,
    feature_ranges=None,
    target_col=None
):
    rows = []

    for instance_idx, cf_row in cf_dict.items():

        if isinstance(cf_row, pd.DataFrame):
            if len(cf_row) == 0:
                continue
            cf_row = cf_row.iloc[0]

        if isinstance(cf_row, dict):
            cf_row = pd.Series(cf_row)

        cf_row = cf_row.copy()

        if target_col is not None:
            cf_row = cf_row.drop(labels=[target_col], errors="ignore")

        residual_results = {}
        total_scm_residual = 0.0
        total_normalized_scm_residual = 0.0

        for feat in endogenous_features:
            if feat not in scm:
                continue

            scm_pred = scm[feat]["func"](cf_row)
            actual = float(cf_row[feat])
            residual = abs(actual - scm_pred)

            residual_results[f"scm_pred_{feat}"] = scm_pred
            residual_results[f"scm_actual_{feat}"] = actual
            residual_results[f"scm_residual_{feat}"] = residual

            total_scm_residual += residual

            if feature_ranges is not None and feat in feature_ranges:
                low, high = feature_ranges[feat]
                feat_range = max(high - low, 1e-6)
                normalized_residual = residual / feat_range
            else:
                normalized_residual = np.nan

            residual_results[f"normalized_scm_residual_{feat}"] = normalized_residual

            if not np.isnan(normalized_residual):
                total_normalized_scm_residual += normalized_residual

        residual_results["total_scm_residual"] = total_scm_residual
        residual_results["total_normalized_scm_residual"] = total_normalized_scm_residual

        rows.append({
            "instance_index": instance_idx,
            **residual_results
        })

    residuals_df = pd.DataFrame(rows)

    if len(residuals_df) == 0:
        summary = {
            "total_cfs_checked": 0,
            "avg_total_scm_residual": np.nan,
            "avg_total_normalized_scm_residual": np.nan,
            "median_total_scm_residual": np.nan,
            "median_total_normalized_scm_residual": np.nan
        }
        return residuals_df, summary

    summary = {
        "total_cfs_checked": len(residuals_df),
        "avg_total_scm_residual": residuals_df["total_scm_residual"].mean(),
        "avg_total_normalized_scm_residual": residuals_df["total_normalized_scm_residual"].mean(),
        "median_total_scm_residual": residuals_df["total_scm_residual"].median(),
        "median_total_normalized_scm_residual": residuals_df["total_normalized_scm_residual"].median()
    }

    return residuals_df, summary



# 8. COMPARE CFs FROM MODEL 1 VS MODEL 2


def compare_cf_dicts_between_models(
    cf_rows_model_1,
    cf_rows_model_2,
    X_test,
    model_1,
    model_2,
    continuous_features,
    categorical_features,
    feature_ranges=None,
    only_same_original_prediction=True,
    tol=1e-6
):
    rows = []

    common_instances = sorted(
        set(cf_rows_model_1.keys()).intersection(set(cf_rows_model_2.keys()))
    )

    for i in common_instances:
        query_instance = X_test.iloc[[int(i)]].copy()

        orig_pred_1 = model_1.predict(query_instance)[0]
        orig_pred_2 = model_2.predict(query_instance)[0]

        if only_same_original_prediction and int(orig_pred_1) != int(orig_pred_2):
            continue

        cf_1 = cf_rows_model_1[i]
        cf_2 = cf_rows_model_2[i]

        if isinstance(cf_1, pd.DataFrame):
            cf_1 = cf_1.iloc[0]
        if isinstance(cf_2, pd.DataFrame):
            cf_2 = cf_2.iloc[0]

        changed_features = []
        numeric_magnitude = 0.0
        normalized_numeric_magnitude = 0.0
        categorical_differences = 0

        for col in cf_1.index:
            if col not in cf_2.index:
                continue

            v1 = cf_1[col]
            v2 = cf_2[col]

            if col in continuous_features:
                diff = abs(float(v1) - float(v2))

                if diff > tol:
                    changed_features.append(col)
                    numeric_magnitude += diff

                    if feature_ranges is not None and col in feature_ranges:
                        low, high = feature_ranges[col]
                        feature_range = max(high - low, 1e-6)
                        normalized_numeric_magnitude += diff / feature_range

            elif col in categorical_features:
                if v1 != v2:
                    changed_features.append(col)
                    categorical_differences += 1

        rows.append({
            "instance": i,
            "orig_pred_model_1": orig_pred_1,
            "orig_pred_model_2": orig_pred_2,
            "n_features_different_between_models": len(changed_features),
            "numeric_magnitude_between_models": numeric_magnitude,
            "normalized_numeric_magnitude_between_models": normalized_numeric_magnitude,
            "categorical_differences_between_models": categorical_differences,
            "different_features_between_models": changed_features
        })

    comparison_df = pd.DataFrame(rows)

    if len(comparison_df) == 0:
        summary = {
            "n_common_comparable_cfs": 0,
            "avg_n_features_different_between_models": np.nan,
            "avg_numeric_magnitude_between_models": np.nan,
            "avg_normalized_numeric_magnitude_between_models": np.nan,
            "avg_categorical_differences_between_models": np.nan
        }
    else:
        summary = {
            "n_common_comparable_cfs": len(comparison_df),
            "avg_n_features_different_between_models": comparison_df["n_features_different_between_models"].mean(),
            "avg_numeric_magnitude_between_models": comparison_df["numeric_magnitude_between_models"].mean(),
            "avg_normalized_numeric_magnitude_between_models": comparison_df["normalized_numeric_magnitude_between_models"].mean(),
            "avg_categorical_differences_between_models": comparison_df["categorical_differences_between_models"].mean()
        }

    return comparison_df, summary



# 9. ONE FULL SEED RUN


def add_context_to_summary(summary, seed, causal_weight, method, model_name, residual_type=None):
    out = dict(summary)
    out["seed"] = seed
    out["causal_weight"] = causal_weight
    out["method"] = method
    out["model_name"] = model_name
    if residual_type is not None:
        out["residual_type"] = residual_type
    return out


def run_one_seed(df, seed, causal_weights):
    set_all_seeds(seed)

    split = split_data(df, seed)

    X_test = split["X_test"]
    original_dtypes = split["original_dtypes"]

    model_1, model_2, model_metrics = train_models(split, seed)
    model_metrics.insert(0, "seed", seed)

    scm_1, _, scm_summary_1 = fit_scm_models(
        split["X_train_initial"],
        model_name="SCM for Model 1"
    )

    scm_2, _, scm_summary_2 = fit_scm_models(
        split["X_train_retrained"],
        model_name="SCM for Model 2"
    )

    scm_summaries = pd.DataFrame([
        {"seed": seed, "model_name": "Model 1", **scm_summary_1},
        {"seed": seed, "model_name": "Model 2", **scm_summary_2},
    ])

    feature_ranges_1 = make_feature_ranges(split["X_train_initial"])
    feature_ranges_2 = make_feature_ranges(split["X_train_retrained"])

    data_dice_1, model_dice_1 = make_dice_objects(
        split["X_train_initial"],
        split["y_train_initial"],
        model_1
    )

    data_dice_2, model_dice_2 = make_dice_objects(
        split["X_train_retrained"],
        split["y_train_retrained"],
        model_2
    )

    # Normal DiCE is run once per seed.
    # It does not depend on causal_weight.
    explainer_1 = DiceGenetic(
        data_interface=data_dice_1,
        model_interface=model_dice_1
    )

    explainer_2 = DiceGenetic(
        data_interface=data_dice_2,
        model_interface=model_dice_2
    )

    normal_results_df_model_1, normal_cf_rows_model_1, normal_generation_summary_1 = run_cf_stability_for_model(
        model=model_1,
        explainer=explainer_1,
        X_test=X_test,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        feature_ranges=feature_ranges_1,
        original_dtypes=original_dtypes,
        target_col=target_col,
        model_name="Model 1 Normal CF",
        method_name="normal",
        seed=seed,
        causal_weight=np.nan,
        n_instances=N_INSTANCES,
        maxiterations=MAXITERATIONS,
        use_private_generate=True,
        stability_K=STABILITY_K,
        stability_sigma=STABILITY_SIGMA,
        scm=scm_1,
        endogenous_features=endogenous_features
    )

    normal_results_df_model_2, normal_cf_rows_model_2, normal_generation_summary_2 = run_cf_stability_for_model(
        model=model_2,
        explainer=explainer_2,
        X_test=X_test,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        feature_ranges=feature_ranges_2,
        original_dtypes=original_dtypes,
        target_col=target_col,
        model_name="Model 2 Normal CF",
        method_name="normal",
        seed=seed,
        causal_weight=np.nan,
        n_instances=N_INSTANCES,
        maxiterations=MAXITERATIONS,
        use_private_generate=True,
        stability_K=STABILITY_K,
        stability_sigma=STABILITY_SIGMA,
        scm=scm_2,
        endogenous_features=endogenous_features
    )

    normal_residuals_model_1, normal_residuals_summary_1 = evaluate_scm_residuals_for_cfs(
        cf_dict=normal_cf_rows_model_1,
        scm=scm_1,
        endogenous_features=endogenous_features,
        feature_ranges=feature_ranges_1,
        target_col=target_col
    )

    normal_residuals_model_2, normal_residuals_summary_2 = evaluate_scm_residuals_for_cfs(
        cf_dict=normal_cf_rows_model_2,
        scm=scm_2,
        endogenous_features=endogenous_features,
        feature_ranges=feature_ranges_2,
        target_col=target_col
    )

    normal_comparison_df, normal_cf_comparison_summary = compare_cf_dicts_between_models(
        cf_rows_model_1=normal_cf_rows_model_1,
        cf_rows_model_2=normal_cf_rows_model_2,
        X_test=X_test,
        model_1=model_1,
        model_2=model_2,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        feature_ranges=feature_ranges_1,
        only_same_original_prediction=True
    )

    all_cf_results = [
        normal_results_df_model_1,
        normal_results_df_model_2
    ]

    all_generation_summaries = [
        normal_generation_summary_1,
        normal_generation_summary_2
    ]

    all_residual_summaries = [
        add_context_to_summary(normal_residuals_summary_1, seed, np.nan, "normal", "Model 1", "normal_cf_scm_residual"),
        add_context_to_summary(normal_residuals_summary_2, seed, np.nan, "normal", "Model 2", "normal_cf_scm_residual"),
    ]

    normal_comparison_summary_row = {
        "seed": seed,
        "causal_weight": np.nan,
        "method": "normal",
        **normal_cf_comparison_summary
    }

    all_comparison_summaries = [normal_comparison_summary_row]
    all_comparison_dfs = []

    if len(normal_comparison_df) > 0:
        normal_comparison_df = normal_comparison_df.copy()
        normal_comparison_df.insert(0, "seed", seed)
        normal_comparison_df.insert(1, "causal_weight", np.nan)
        normal_comparison_df.insert(2, "method", "normal")
        all_comparison_dfs.append(normal_comparison_df)

    # Causal DiCE is run for every causal weight.
    for causal_weight in causal_weights:
        print("\n\n" + "=" * 90)
        print(f"STARTING CAUSAL WEIGHT {causal_weight} FOR SEED {seed}")
        print("=" * 90)

        causal_explainer_1 = DiceGeneticCausal(
            data_interface=data_dice_1,
            model_interface=model_dice_1,
            scm=scm_1,
            exogenous_features=exogenous_features,
            endogenous_features=endogenous_features,
            causal_weight=causal_weight
        )

        causal_explainer_2 = DiceGeneticCausal(
            data_interface=data_dice_2,
            model_interface=model_dice_2,
            scm=scm_2,
            exogenous_features=exogenous_features,
            endogenous_features=endogenous_features,
            causal_weight=causal_weight
        )

        causal_results_df_model_1, causal_cf_rows_model_1, causal_generation_summary_1 = run_cf_stability_for_model(
            model=model_1,
            explainer=causal_explainer_1,
            X_test=X_test,
            continuous_features=continuous_features,
            categorical_features=categorical_features,
            feature_ranges=feature_ranges_1,
            original_dtypes=original_dtypes,
            target_col=target_col,
            model_name="Model 1 Causal CF",
            method_name="causal",
            seed=seed,
            causal_weight=causal_weight,
            n_instances=N_INSTANCES,
            maxiterations=MAXITERATIONS,
            use_private_generate=True,
            stability_K=STABILITY_K,
            stability_sigma=STABILITY_SIGMA,
            scm=scm_1,
            endogenous_features=endogenous_features
        )

        causal_results_df_model_2, causal_cf_rows_model_2, causal_generation_summary_2 = run_cf_stability_for_model(
            model=model_2,
            explainer=causal_explainer_2,
            X_test=X_test,
            continuous_features=continuous_features,
            categorical_features=categorical_features,
            feature_ranges=feature_ranges_2,
            original_dtypes=original_dtypes,
            target_col=target_col,
            model_name="Model 2 Causal CF",
            method_name="causal",
            seed=seed,
            causal_weight=causal_weight,
            n_instances=N_INSTANCES,
            maxiterations=MAXITERATIONS,
            use_private_generate=True,
            stability_K=STABILITY_K,
            stability_sigma=STABILITY_SIGMA,
            scm=scm_2,
            endogenous_features=endogenous_features
        )

        causal_residuals_model_1, causal_residuals_summary_1 = evaluate_scm_residuals_for_cfs(
            cf_dict=causal_cf_rows_model_1,
            scm=scm_1,
            endogenous_features=endogenous_features,
            feature_ranges=feature_ranges_1,
            target_col=target_col
        )

        causal_residuals_model_2, causal_residuals_summary_2 = evaluate_scm_residuals_for_cfs(
            cf_dict=causal_cf_rows_model_2,
            scm=scm_2,
            endogenous_features=endogenous_features,
            feature_ranges=feature_ranges_2,
            target_col=target_col
        )

        causal_comparison_df, causal_cf_comparison_summary = compare_cf_dicts_between_models(
            cf_rows_model_1=causal_cf_rows_model_1,
            cf_rows_model_2=causal_cf_rows_model_2,
            X_test=X_test,
            model_1=model_1,
            model_2=model_2,
            continuous_features=continuous_features,
            categorical_features=categorical_features,
            feature_ranges=feature_ranges_1,
            only_same_original_prediction=True
        )

        all_cf_results.extend([
            causal_results_df_model_1,
            causal_results_df_model_2
        ])

        all_generation_summaries.extend([
            causal_generation_summary_1,
            causal_generation_summary_2
        ])

        all_residual_summaries.extend([
            add_context_to_summary(causal_residuals_summary_1, seed, causal_weight, "causal", "Model 1", "causal_cf_scm_residual"),
            add_context_to_summary(causal_residuals_summary_2, seed, causal_weight, "causal", "Model 2", "causal_cf_scm_residual"),
        ])

        all_comparison_summaries.append({
            "seed": seed,
            "causal_weight": causal_weight,
            "method": "causal",
            **causal_cf_comparison_summary
        })

        if len(causal_comparison_df) > 0:
            causal_comparison_df = causal_comparison_df.copy()
            causal_comparison_df.insert(0, "seed", seed)
            causal_comparison_df.insert(1, "causal_weight", causal_weight)
            causal_comparison_df.insert(2, "method", "causal")
            all_comparison_dfs.append(causal_comparison_df)

    seed_outputs = {
        "model_metrics": model_metrics,
        "scm_summaries": scm_summaries,
        "cf_results": pd.concat(all_cf_results, ignore_index=True) if len(all_cf_results) else pd.DataFrame(),
        "generation_summaries": pd.DataFrame(all_generation_summaries),
        "residual_summaries": pd.DataFrame(all_residual_summaries),
        "comparison_summaries": pd.DataFrame(all_comparison_summaries),
        "comparison_details": pd.concat(all_comparison_dfs, ignore_index=True) if len(all_comparison_dfs) else pd.DataFrame()
    }

    return seed_outputs


# 10. FULL GRID RUN

def run_full_experiment():
    # Important:
    # If DATA_PATH is None, generate the dataset ONCE.
    # Then all seeds use different splits/training randomness on the same dataset.
    # This matches your German setup more closely than regenerating a new dataset
    # inside every seed.
    df = load_data(DATA_PATH, seed=42)

    all_model_metrics = []
    all_scm_summaries = []
    all_cf_results = []
    all_generation_summaries = []
    all_residual_summaries = []
    all_comparison_summaries = []
    all_comparison_details = []

    for seed in SEEDS:
        print(f"STARTING FULL EXPERIMENT FOR THIS SEED {seed}") # JUST TO TRACK RUNNING TIME

        seed_outputs = run_one_seed(
            df=df,
            seed=seed,
            causal_weights=CAUSAL_WEIGHTS
        )

        all_model_metrics.append(seed_outputs["model_metrics"])
        all_scm_summaries.append(seed_outputs["scm_summaries"])
        all_cf_results.append(seed_outputs["cf_results"])
        all_generation_summaries.append(seed_outputs["generation_summaries"])
        all_residual_summaries.append(seed_outputs["residual_summaries"])
        all_comparison_summaries.append(seed_outputs["comparison_summaries"])

        if len(seed_outputs["comparison_details"]) > 0:
            all_comparison_details.append(seed_outputs["comparison_details"])

        # Save partial files after each seed so you do not lose progress.
        partial_prefix = os.path.join(OUTPUT_DIR, f"partial_after_seed_{seed}")

        pd.concat(all_model_metrics, ignore_index=True).to_csv(
            f"{partial_prefix}_model_metrics.csv", index=False
        )
        pd.concat(all_scm_summaries, ignore_index=True).to_csv(
            f"{partial_prefix}_scm_summaries.csv", index=False
        )
        pd.concat(all_cf_results, ignore_index=True).to_csv(
            f"{partial_prefix}_cf_results.csv", index=False
        )
        pd.concat(all_generation_summaries, ignore_index=True).to_csv(
            f"{partial_prefix}_generation_summaries.csv", index=False
        )
        pd.concat(all_residual_summaries, ignore_index=True).to_csv(
            f"{partial_prefix}_residual_summaries.csv", index=False
        )
        pd.concat(all_comparison_summaries, ignore_index=True).to_csv(
            f"{partial_prefix}_comparison_summaries.csv", index=False
        )

        if len(all_comparison_details) > 0:
            pd.concat(all_comparison_details, ignore_index=True).to_csv(
                f"{partial_prefix}_comparison_details.csv", index=False
            )

    final_outputs = {
        "model_metrics": pd.concat(all_model_metrics, ignore_index=True),
        "scm_summaries": pd.concat(all_scm_summaries, ignore_index=True),
        "cf_results": pd.concat(all_cf_results, ignore_index=True),
        "generation_summaries": pd.concat(all_generation_summaries, ignore_index=True),
        "residual_summaries": pd.concat(all_residual_summaries, ignore_index=True),
        "comparison_summaries": pd.concat(all_comparison_summaries, ignore_index=True),
        "comparison_details": pd.concat(all_comparison_details, ignore_index=True) if len(all_comparison_details) > 0 else pd.DataFrame()
    }

    final_outputs["model_metrics"].to_csv(
        os.path.join(OUTPUT_DIR, "final_model_metrics.csv"), index=False
    )
    final_outputs["scm_summaries"].to_csv(
        os.path.join(OUTPUT_DIR, "final_scm_summaries.csv"), index=False
    )
    final_outputs["cf_results"].to_csv(
        os.path.join(OUTPUT_DIR, "final_cf_results.csv"), index=False
    )
    final_outputs["generation_summaries"].to_csv(
        os.path.join(OUTPUT_DIR, "final_generation_summaries.csv"), index=False
    )
    final_outputs["residual_summaries"].to_csv(
        os.path.join(OUTPUT_DIR, "final_residual_summaries.csv"), index=False
    )
    final_outputs["comparison_summaries"].to_csv(
        os.path.join(OUTPUT_DIR, "final_comparison_summaries.csv"), index=False
    )
    final_outputs["comparison_details"].to_csv(
        os.path.join(OUTPUT_DIR, "final_comparison_details.csv"), index=False
    )

    return final_outputs


# 11. SUMMARY TABLES

def make_summary_tables(final_outputs):
    model_metrics = final_outputs["model_metrics"]
    cf_results = final_outputs["cf_results"]
    generation_summaries = final_outputs["generation_summaries"]
    residual_summaries = final_outputs["residual_summaries"]
    comparison_summaries = final_outputs["comparison_summaries"]
    scm_summaries = final_outputs["scm_summaries"]

    model_summary = (
        model_metrics
        .groupby("model")
        .agg({
            "accuracy": ["mean", "std"],
            "f1": ["mean", "std"],
            "precision": ["mean", "std"],
            "recall": ["mean", "std"],
            "roc_auc": ["mean", "std"]
        })
    )

    scm_summary = (
        scm_summaries
        .groupby("model_name")
        .agg({
            "scm_r2_true_equation_to_x3": ["mean", "std"],
            "scm_residual_mean": ["mean", "std"],
            "scm_residual_std": ["mean", "std"]
        })
    )

    generation_summary = (
        generation_summaries
        .groupby(["method", "causal_weight", "model"], dropna=False)
        .agg({
            "n_attempted": ["mean", "std"],
            "n_valid_cfs": ["mean", "std"],
            "n_no_cf_found": ["mean", "std"],
            "n_invalid_cfs": ["mean", "std"],
            "n_exceptions": ["mean", "std"],
            "success_rate": ["mean", "std"]
        })
    )

    cf_metric_cols = [
        "cf_prob_desired",
        "stability_score",
        "mean_neighbourhood_prob",
        "std_neighbourhood_prob",
        "min_neighbourhood_prob",
        "max_neighbourhood_prob",
        "validity_in_neighbourhood",
        "n_changed_features",
        "numeric_change_sum",
        "normalized_numeric_change_sum",
        "categorical_change_count",
        "scm_violation_sum",
        "normalized_scm_violation_sum"
    ]

    cf_metric_cols = [c for c in cf_metric_cols if c in cf_results.columns]

    cf_summary = (
        cf_results
        .groupby(["method", "causal_weight", "model_name"], dropna=False)[cf_metric_cols]
        .agg(["mean", "std", "median"])
    )

    residual_summary = (
        residual_summaries
        .groupby(["method", "causal_weight", "model_name"], dropna=False)
        .agg({
            "total_cfs_checked": ["mean", "std"],
            "avg_total_scm_residual": ["mean", "std"],
            "avg_total_normalized_scm_residual": ["mean", "std"],
            "median_total_scm_residual": ["mean", "std"],
            "median_total_normalized_scm_residual": ["mean", "std"]
        })
    )

    comparison_summary = (
        comparison_summaries
        .groupby(["method", "causal_weight"], dropna=False)
        .agg({
            "n_common_comparable_cfs": ["mean", "std"],
            "avg_n_features_different_between_models": ["mean", "std"],
            "avg_numeric_magnitude_between_models": ["mean", "std"],
            "avg_normalized_numeric_magnitude_between_models": ["mean", "std"],
            "avg_categorical_differences_between_models": ["mean", "std"]
        })
    )

    summary_tables = {
        "model_summary": model_summary,
        "scm_summary": scm_summary,
        "generation_summary": generation_summary,
        "cf_summary": cf_summary,
        "residual_summary": residual_summary,
        "comparison_summary": comparison_summary
    }

    model_summary.to_csv(os.path.join(OUTPUT_DIR, "summary_model_metrics_mean_std.csv"))
    scm_summary.to_csv(os.path.join(OUTPUT_DIR, "summary_scm_mean_std.csv"))
    generation_summary.to_csv(os.path.join(OUTPUT_DIR, "summary_generation_mean_std.csv"))
    cf_summary.to_csv(os.path.join(OUTPUT_DIR, "summary_cf_metrics_mean_std_median.csv"))
    residual_summary.to_csv(os.path.join(OUTPUT_DIR, "summary_residuals_mean_std.csv"))
    comparison_summary.to_csv(os.path.join(OUTPUT_DIR, "summary_comparison_mean_std.csv"))

    return summary_tables


if __name__ == "__main__":
    final_outputs = run_full_experiment()
    summary_tables = make_summary_tables(final_outputs)

    print("\n\nDONE. Files saved in:")
    print(OUTPUT_DIR)
