# Credit Risk Prediction Pipeline

End-to-end ML pipeline for predicting loan default (`Charged Off` vs `Fully Paid`).

## Data

| File | Rows | Notes |
|------|------|-------|
| `Datasets/credit_train.csv` | 100,000 | Labeled (`Loan Status`) |
| `Datasets/credit_test.csv` | 10,353 | Unlabeled (353 empty rows dropped → 10,000) |

**Target:** `Loan Status` → binary `1 = Charged Off`, `0 = Fully Paid`  
**Class balance (after dedup):** ~75% Fully Paid / 25% Charged Off

## Pipeline stages

```
1. Load
2. Clean
   - Drop empty rows & ID columns
   - Fix Credit Score outliers (>850 → /10)
   - Replace Loan Amount sentinel 99999999 → NaN
   - Normalize Home Ownership / Purpose
   - Ordinal-encode Years in current job
   - Drop duplicate rows (~18k in train)
3. Feature engineering
   - DTI  (Monthly Debt × 12 / Annual Income)
   - Credit_Util  (Balance / Max Open Credit, capped at 2)
   - LTI  (Loan Amount / Annual Income)
   - Has_Delinquency flag
   - Binary flags for credit problems / bankruptcies / tax liens
4. Preprocess
   - Median impute + StandardScaler (numeric)
   - Mode impute + OneHotEncoder (categorical)
5. Train & evaluate
   - Logistic Regression (balanced)
   - Random Forest (balanced_subsample)
   - Gradient Boosting
6. Predict test set with best model (by ROC-AUC)
```

## Usage

```bash
cd Kernels/credit_pipeline
python3 pipeline.py
```

## Outputs (`outputs/`)

| File | Description |
|------|-------------|
| `model_comparison.csv` | Validation metrics for all models |
| `test_predictions.csv` | Loan ID, Customer ID, Predicted_Default, Default_Probability |
| `best_model_*.pkl` | Serialized sklearn Pipeline (preprocessor + classifier) |

## Validation results (hold-out 20%)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Gradient Boosting | 0.75 | 0.56 | 0.07 | 0.13 | **0.714** |
| Random Forest | 0.64 | 0.37 | 0.67 | 0.48 | 0.709 |
| Logistic Regression | 0.62 | 0.35 | 0.58 | 0.44 | 0.647 |

> **Note:** Gradient Boosting maximizes ROC-AUC but has low recall on the minority class.  
> Random Forest offers a better precision–recall trade-off for screening high-risk loans.  
> Choose the operating threshold on the probability scores according to business cost of false positives vs false negatives.
