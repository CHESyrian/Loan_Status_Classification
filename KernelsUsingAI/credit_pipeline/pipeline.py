"""
Credit Risk Prediction Pipeline
================================
End-to-end pipeline for loan default prediction on the Credit dataset.

Target: Loan Status (Fully Paid vs Charged Off)
Train:  credit_train.csv  (~100k rows)
Test:   credit_test.csv   (~10k rows, no label)

Pipeline stages
---------------
1. Load & clean
2. Feature engineering
3. Preprocess (impute, encode, scale)
4. Train / evaluate models
5. Generate test predictions
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # /home/workdir/artifacts
DATA_DIR = ROOT / "Datasets"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "credit_train.csv"
TEST_PATH = DATA_DIR / "credit_test.csv"

RANDOM_STATE = 42
TARGET = "Loan Status"
ID_COLS = ["Loan ID", "Customer ID"]

# ---------------------------------------------------------------------------
# 1. Load & Clean
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


def clean(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Apply dataset-specific cleaning rules."""
    df = df.copy()

    # Drop completely empty rows (present in test)
    df = df.dropna(how="all")

    # Drop ID columns (not predictive; high cardinality)
    df = df.drop(columns=[c for c in ID_COLS if c in df.columns], errors="ignore")

    # --- Current Loan Amount: 99999999 is a sentinel for "unknown" ---
    if "Current Loan Amount" in df.columns:
        df["Current Loan Amount"] = df["Current Loan Amount"].replace(99999999, np.nan)

    # --- Credit Score: values > 850 are typically true_score * 10 ---
    if "Credit Score" in df.columns:
        mask = df["Credit Score"] > 850
        df.loc[mask, "Credit Score"] = df.loc[mask, "Credit Score"] / 10

    # --- Home Ownership: normalise HaveMortgage ---
    if "Home Ownership" in df.columns:
        df["Home Ownership"] = df["Home Ownership"].replace(
            {"HaveMortgage": "Home Mortgage"}
        )

    # --- Purpose: normalise casing ---
    if "Purpose" in df.columns:
        df["Purpose"] = df["Purpose"].str.lower().str.replace(" ", "_")

    # --- Years in current job: map to ordinal numeric ---
    if "Years in current job" in df.columns:
        job_map = {
            "< 1 year": 0,
            "1 year": 1,
            "2 years": 2,
            "3 years": 3,
            "4 years": 4,
            "5 years": 5,
            "6 years": 6,
            "7 years": 7,
            "8 years": 8,
            "9 years": 9,
            "10+ years": 10,
            "n/a": np.nan,
        }
        df["Years in current job"] = df["Years in current job"].map(job_map)

    # --- Target encoding (train only) ---
    if is_train and TARGET in df.columns:
        df[TARGET] = (df[TARGET] == "Charged Off").astype(int)  # 1 = default

    # Drop exact duplicate rows (train has ~18k duplicate Loan IDs)
    df = df.drop_duplicates()

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Debt-to-Income ratio
    if {"Monthly Debt", "Annual Income"}.issubset(df.columns):
        df["DTI"] = np.where(
            df["Annual Income"] > 0,
            (df["Monthly Debt"] * 12) / df["Annual Income"],
            np.nan,
        )

    # Credit utilisation
    if {"Current Credit Balance", "Maximum Open Credit"}.issubset(df.columns):
        df["Credit_Util"] = np.where(
            df["Maximum Open Credit"] > 0,
            df["Current Credit Balance"] / df["Maximum Open Credit"],
            np.nan,
        )
        # Cap extreme utilisation at 2.0
        df["Credit_Util"] = df["Credit_Util"].clip(upper=2.0)

    # Loan-to-Income ratio
    if {"Current Loan Amount", "Annual Income"}.issubset(df.columns):
        df["LTI"] = np.where(
            df["Annual Income"] > 0,
            df["Current Loan Amount"] / df["Annual Income"],
            np.nan,
        )

    # Has history of delinquency (binary from months since last delinquent)
    if "Months since last delinquent" in df.columns:
        df["Has_Delinquency"] = df["Months since last delinquent"].notna().astype(int)
        # Fill remaining NaN with a large sentinel so tree models can split
        # (will be imputed for linear models via the preprocessor)
        df["Months since last delinquent"] = df["Months since last delinquent"].fillna(-1)

    # Binary flags for credit problems / bankruptcies / tax liens
    for col in ["Number of Credit Problems", "Bankruptcies", "Tax Liens"]:
        if col in df.columns:
            df[f"Has_{col.replace(' ', '_')}"] = (df[col].fillna(0) > 0).astype(int)

    return df


# ---------------------------------------------------------------------------
# 3. Preprocessing pipeline
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "Current Loan Amount",
    "Credit Score",
    "Annual Income",
    "Years in current job",
    "Monthly Debt",
    "Years of Credit History",
    "Months since last delinquent",
    "Number of Open Accounts",
    "Number of Credit Problems",
    "Current Credit Balance",
    "Maximum Open Credit",
    "Bankruptcies",
    "Tax Liens",
    "DTI",
    "Credit_Util",
    "LTI",
]

BINARY_FEATURES = [
    "Has_Delinquency",
    "Has_Number_of_Credit_Problems",
    "Has_Bankruptcies",
    "Has_Tax_Liens",
]

CATEGORICAL_FEATURES = [
    "Term",
    "Home Ownership",
    "Purpose",
]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    binary_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
            ("bin", binary_pipe, BINARY_FEATURES),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# 4. Models
# ---------------------------------------------------------------------------

def get_models() -> dict[str, Any]:
    return {
        "logistic": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=30,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_model(
    name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, float]:
    pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", model),
        ]
    )
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_val)
    y_proba = pipe.predict_proba(X_val)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_val, y_proba),
    }

    print(f"\n{'='*50}")
    print(f"  {name.upper()}")
    print(f"{'='*50}")
    print(classification_report(y_val, y_pred, target_names=["Fully Paid", "Charged Off"]))
    print("Confusion matrix:\n", confusion_matrix(y_val, y_pred))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    return metrics, pipe


# ---------------------------------------------------------------------------
# 5. Main orchestration
# ---------------------------------------------------------------------------

def run_pipeline(save_artifacts: bool = True) -> dict[str, Any]:
    print("=" * 60)
    print("  CREDIT RISK PREDICTION PIPELINE")
    print("=" * 60)

    # --- Load ---
    print("\n[1/5] Loading data...")
    train_raw, test_raw = load_data()
    print(f"  Train raw: {train_raw.shape}  |  Test raw: {test_raw.shape}")

    # --- Clean ---
    print("\n[2/5] Cleaning...")
    train = clean(train_raw, is_train=True)
    test = clean(test_raw, is_train=False)
    print(f"  Train clean: {train.shape}  |  Test clean: {test.shape}")
    print(f"  Target distribution:\n{train[TARGET].value_counts(normalize=True).round(3)}")

    # --- Feature engineering ---
    print("\n[3/5] Feature engineering...")
    train = engineer_features(train)
    test = engineer_features(test)

    feature_cols = [
        c for c in NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
        if c in train.columns
    ]
    X = train[feature_cols]
    y = train[TARGET]
    X_test = test[[c for c in feature_cols if c in test.columns]]

    # Align test columns (in case any are missing)
    for c in feature_cols:
        if c not in X_test.columns:
            X_test[c] = np.nan
    X_test = X_test[feature_cols]

    # --- Train / validation split ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  Train split: {X_train.shape}  |  Val split: {X_val.shape}")

    # --- Train & evaluate models ---
    print("\n[4/5] Training models...")
    results = []
    fitted_pipes = {}

    for name, model in get_models().items():
        metrics, pipe = evaluate_model(name, model, X_train, y_train, X_val, y_val)
        results.append(metrics)
        fitted_pipes[name] = pipe

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON (validation set)")
    print("=" * 60)
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    best_pipe = fitted_pipes[best_name]
    print(f"\n>>> Best model: {best_name}  (ROC-AUC = {results_df.iloc[0]['roc_auc']:.4f})")

    # --- Refit best model on full training data ---
    print("\n[5/5] Refitting best model on full training data & predicting test...")
    best_pipe.fit(X, y)

    test_proba = best_pipe.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)

    # Recover test IDs from original non-empty rows
    test_clean_ids = (
        test_raw.dropna(how="all")
        .drop_duplicates(subset=["Loan ID"] if "Loan ID" in test_raw.columns else None)
        .reset_index(drop=True)
    )
    n = min(len(test_clean_ids), len(test_pred))
    submission = pd.DataFrame(
        {
            "Loan ID": test_clean_ids["Loan ID"].iloc[:n].values,
            "Customer ID": test_clean_ids["Customer ID"].iloc[:n].values,
            "Predicted_Default": test_pred[:n],
            "Default_Probability": np.round(test_proba[:n], 4),
        }
    )

    if save_artifacts:
        results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
        submission.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
        with open(OUTPUT_DIR / f"best_model_{best_name}.pkl", "wb") as f:
            pickle.dump(best_pipe, f)
        print(f"\n  Artifacts saved to {OUTPUT_DIR}/")
        print(f"    - model_comparison.csv")
        print(f"    - test_predictions.csv")
        print(f"    - best_model_{best_name}.pkl")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)

    return {
        "results": results_df,
        "best_model_name": best_name,
        "best_pipe": best_pipe,
        "submission": submission,
    }


if __name__ == "__main__":
    run_pipeline(save_artifacts=True)
