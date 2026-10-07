#!/usr/bin/env python3
"""
Production scoring for credit risk model.

Usage:
  python score_loans.py --input path/to/loans.csv --output path/to/scored.csv
  python score_loans.py --input path/to/loans.csv --output scored.csv --threshold 0.27

Expects raw columns as in credit_test.csv (no Loan Status required).
Applies: clean → feature engineering → Standard-25 alignment → model predict.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# Default locations relative to this script
HERE = Path(__file__).resolve().parent
FEAT_DIR = HERE / "features"
OUT_DIR = HERE / "model_outputs"

JOB_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10, "n/a": np.nan,
}
PURPOSE_GROUPS = {
    "debt_consolidation": "debt_consolidation", "home_improvements": "home",
    "buy_house": "home", "buy_a_car": "vehicle", "business_loan": "business",
    "small_business": "business", "medical_bills": "medical",
    "take_a_trip": "discretionary", "vacation": "discretionary",
    "wedding": "discretionary", "major_purchase": "discretionary",
    "moving": "discretionary", "educational_expenses": "education",
    "renewable_energy": "other", "other": "other",
}


def engineer(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.dropna(how="all").copy()
    if "Loan ID" in d.columns:
        d = d.drop_duplicates(subset=["Loan ID"])
    if "Credit Score" in d.columns:
        m = d["Credit Score"] > 850
        d.loc[m, "Credit Score"] = d.loc[m, "Credit Score"] / 10
    if "Current Loan Amount" in d.columns:
        d["Current Loan Amount"] = d["Current Loan Amount"].replace(99999999, np.nan)
    if "Home Ownership" in d.columns:
        d["Home Ownership"] = d["Home Ownership"].replace({"HaveMortgage": "Home Mortgage"})
    if "Purpose" in d.columns:
        d["Purpose"] = d["Purpose"].astype(str).str.lower().str.strip().str.replace(" ", "_")

    d["job_years"] = d["Years in current job"].map(JOB_MAP) if "Years in current job" in d.columns else np.nan
    d["is_short_tenure"] = (pd.Series(d["job_years"]).fillna(99) <= 1).astype(int)

    for col, alias in [
        ("Credit Score", "credit_score"),
        ("Annual Income", "annual_income"),
        ("Current Loan Amount", "loan_amount"),
    ]:
        if col in d.columns:
            d[f"{alias}_missing"] = d[col].isna().astype(int)

    if "Monthly Debt" in d.columns and "Annual Income" in d.columns:
        d["dti"] = np.where(d["Annual Income"] > 0, (d["Monthly Debt"] * 12) / d["Annual Income"], np.nan)
        d["debt_to_loan"] = np.where(
            d["Current Loan Amount"] > 0, d["Monthly Debt"] / d["Current Loan Amount"], np.nan
        )
    if "Annual Income" in d.columns and "Current Loan Amount" in d.columns:
        d["lti"] = np.where(d["Annual Income"] > 0, d["Current Loan Amount"] / d["Annual Income"], np.nan)
    if "Current Credit Balance" in d.columns and "Maximum Open Credit" in d.columns:
        cu = np.where(
            d["Maximum Open Credit"] > 0,
            d["Current Credit Balance"] / d["Maximum Open Credit"],
            np.nan,
        )
        d["credit_util"] = pd.Series(cu, index=d.index).clip(upper=2.0)
        d["balance_to_income"] = np.where(
            d["Annual Income"] > 0, d["Current Credit Balance"] / d["Annual Income"], np.nan
        )

    d["is_long_term"] = (d["Term"] == "Long Term").astype(int) if "Term" in d.columns else 0
    d["is_renter"] = (d["Home Ownership"] == "Rent").astype(int) if "Home Ownership" in d.columns else 0
    d["is_mortgage"] = (d["Home Ownership"] == "Home Mortgage").astype(int) if "Home Ownership" in d.columns else 0
    d["longterm_x_renter"] = d["is_long_term"] * d["is_renter"]
    d["short_tenure_x_longterm"] = d["is_short_tenure"] * d["is_long_term"]

    if "Purpose" in d.columns:
        d["purpose_group"] = d["Purpose"].map(PURPOSE_GROUPS).fillna("other")
        d["purpose_business"] = (d["purpose_group"] == "business").astype(int)

    def qbin(s, q=5):
        try:
            return pd.qcut(s, q=q, duplicates="drop").cat.codes.replace(-1, np.nan)
        except Exception:
            return pd.Series(np.nan, index=s.index)

    if "Credit Score" in d.columns:
        d["credit_score_bin"] = qbin(d["Credit Score"])

        def score_band(s):
            if pd.isna(s):
                return "missing"
            if s < 650:
                return "poor"
            if s < 700:
                return "fair"
            if s < 740:
                return "good"
            return "excellent"

        d["credit_score_band"] = d["Credit Score"].apply(score_band)
        for band in ["poor", "fair", "good", "excellent", "missing"]:
            d[f"score_band_{band}"] = (d["credit_score_band"] == band).astype(int)

    if "Annual Income" in d.columns:
        d["income_bin"] = qbin(d["Annual Income"])
    if "Maximum Open Credit" in d.columns:
        d["log_max_open_credit"] = np.log1p(d["Maximum Open Credit"].clip(lower=0))

    return d


def load_model_and_config(model_path: Path | None = None):
    # Prefer calibrated model if present
    cal = OUT_DIR / "best_model_calibrated.pkl"
    hparam = OUT_DIR / "best_model_hparam_gbm_default.pkl"
    legacy = OUT_DIR / "best_model_gradient_boosting.pkl"
    path = model_path
    if path is None:
        for candidate in (cal, hparam, legacy):
            if candidate.exists():
                path = candidate
                break
    if path is None or not Path(path).exists():
        raise FileNotFoundError("No model pickle found under model_outputs/")

    with open(path, "rb") as f:
        model = pickle.load(f)

    feat_list = FEAT_DIR / "selected_features_standard.json"
    with open(feat_list) as f:
        features = json.load(f)["features"]

    threshold = 0.27
    meta_cal = OUT_DIR / "calibration_threshold_meta.json"
    meta_hp = OUT_DIR / "hparam_search_summary.json"
    if meta_cal.exists():
        threshold = float(json.loads(meta_cal.read_text()).get("threshold_primary", threshold))
    elif meta_hp.exists():
        threshold = float(json.loads(meta_hp.read_text()).get("threshold", threshold))

    return model, features, threshold, str(path)


def score_file(input_path: str, output_path: str, threshold: float | None = None, model_path: str | None = None):
    model, features, default_thr, used_model = load_model_and_config(
        Path(model_path) if model_path else None
    )
    thr = default_thr if threshold is None else float(threshold)

    raw = pd.read_csv(input_path)
    eng = engineer(raw)
    for c in features:
        if c not in eng.columns:
            eng[c] = np.nan
    X = eng[features]

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= thr).astype(int)

    out = pd.DataFrame()
    if "Loan ID" in eng.columns:
        out["Loan ID"] = eng["Loan ID"].values
    if "Customer ID" in eng.columns:
        out["Customer ID"] = eng["Customer ID"].values
    out["Default_Probability"] = np.round(proba, 4)
    out["Predicted_Default"] = pred
    out["Threshold_Used"] = thr

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Model: {used_model}")
    print(f"Scored {len(out):,} rows → {output_path}")
    print(f"Threshold: {thr:.2f} | Flag rate: {pred.mean()*100:.1f}%")
    return out


def main():
    p = argparse.ArgumentParser(description="Score credit loans with the champion default model")
    p.add_argument("--input", "-i", required=True, help="Input CSV (raw loan features)")
    p.add_argument("--output", "-o", required=True, help="Output scored CSV")
    p.add_argument("--threshold", "-t", type=float, default=None, help="Override decision threshold")
    p.add_argument("--model", "-m", default=None, help="Optional path to model .pkl")
    args = p.parse_args()
    score_file(args.input, args.output, args.threshold, args.model)


if __name__ == "__main__":
    main()
