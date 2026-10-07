# Credit Risk Default Prediction

End-to-end analytics project for predicting **loan default** (Charged Off vs Fully Paid) from application and credit-bureau style features.

---

## 1. Project overview

| Item | Detail |
|------|--------|
| **Objective** | Rank personal loan applications by probability of default and support underwriting decisions |
| **Target** | `Default` = 1 if `Loan Status` = *Charged Off*, else 0 |
| **Champion model** | Gradient Boosting Classifier (`n_estimators=80`, `max_depth=4`, `learning_rate=0.1`, `subsample=0.8`) |
| **Feature set** | **Standard tier (25 features)** after engineering + selection |
| **Validation ROC-AUC** | **~0.725** |
| **Recommended threshold** | **0.27** (max F1); cost-based alternative **0.17** when FN ≫ FP |

---

## 2. Repository layout

```
artifacts/
├── Datasets/
│   ├── credit_train.csv          # Labeled training data
│   └── credit_test.csv           # Unlabeled score set
├── Kernels/
│   └── Analytics/
│       ├── 01_credit_train_eda.ipynb
│       ├── 02_credit_feature_engineering.ipynb
│       ├── 03_credit_feature_selection.ipynb
│       ├── 04_credit_modelling.ipynb
│       ├── 05_model_card.ipynb
│       ├── 06_calibration_and_threshold.ipynb
│       ├── 07_error_analysis.ipynb
│       ├── score_loans.py        # Production scoring CLI
│       ├── features/             # Engineered matrices + selection artifacts
│       └── model_outputs/        # Models, metrics, predictions, plots
├── requirements.txt
└── README.md
```

---

## 3. Data

### Training (`credit_train.csv`)
- ~100k raw rows; **~82k** after deduplicating on `Loan ID`
- Default rate ≈ **27.6%**
- Mix of application fields (term, purpose, home ownership, job tenure) and credit attributes (score, income, balances, delinquencies)

### Test (`credit_test.csv`)
- **10,000** applications without labels — scored for production-style output

### Data quality issues addressed
| Issue | Treatment |
|-------|-----------|
| Duplicate `Loan ID` | Keep first occurrence |
| Credit scores > 850 | Interpret as score × 10 and divide by 10 |
| Loan amount = `99999999` | Treat as missing |
| `HaveMortgage` | Map to `Home Mortgage` |
| Inconsistent `Purpose` labels | Lowercase / normalise / group rare levels |

---

## 4. Analysis pipeline

### Step 1 — EDA (`01_credit_train_eda.ipynb`)
- Profile missingness, distributions, target balance
- Bivariate links: term, credit score, DTI-style ratios, home ownership
- Document cleaning contract used in all later stages

### Step 2 — Feature engineering (`02_credit_feature_engineering.ipynb`)
Built interpretable features, including:
- **Ratios:** DTI, LTI, credit utilisation, debt-to-loan, balance-to-income  
- **Flags:** long-term, renter, short tenure, public-record proxies, purpose groups  
- **Missingness indicators** (especially loan amount / score / income)  
- **Bins & score bands**, log transforms, selected interactions  

Outputs under `Kernels/Analytics/features/`.

### Step 3 — Feature selection (`03_credit_feature_selection.ipynb`)
Gates applied:
1. Drop constant / rare binary / ultra-sparse fields  
2. Rank by Pearson correlation + mutual information  
3. Drop one of each highly correlated pair (\|r\| ≥ 0.90)  
4. Random Forest importance + **consensus rank**  

**Recommended set: Standard (25 features)** — similar AUC to the full filtered set with less complexity.

### Step 4 — Modelling (`04_credit_modelling.ipynb`)
Compared on the same Standard features:

| Model | Val ROC-AUC (approx.) |
|-------|------------------------|
| Gradient Boosting | **0.725–0.728** |
| Random Forest | ~0.724 |
| Logistic Regression | ~0.718 |

GBM chosen as primary ranking model; threshold tuned away from 0.5 because of class imbalance.

### Step 5 — Hyperparameter search
Randomized search (sample-based for runtime) did **not** materially beat GBM defaults on full validation AUC. Tuned RF was nearly tied and stronger on recall at threshold 0.5.

### Step 6 — Model card (`05_model_card.ipynb`)
Documents intended use, limits, metrics, feature list, and monitoring suggestions.

### Step 7 — Calibration & threshold (`06_calibration_and_threshold.ipynb`)
- Sigmoid calibration slightly improved Brier score  
- **Max-F1 threshold ≈ 0.27** (balanced screening)  
- **Cost-based threshold ≈ 0.17** if false negatives cost ~5× false positives (high recall, higher flag rate)

### Step 8 — Error analysis (`07_error_analysis.ipynb`)
- Long-term and poor-score segments: high capture of defaults at aggressive thresholds  
- **Excellent credit band:** higher miss rate among actual defaults — stress factors (LTI/DTI, term) still matter  
- `loan_amount_missing` is a strong signal in training; monitor for stability on new vintages  

---

## 5. How to run

### Environment
```bash
cd artifacts
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Notebooks
Open and run in order under `Kernels/Analytics/` (01 → 07). Working directory for notebooks should be `Kernels/Analytics` so relative paths to `features/` and `model_outputs/` resolve.

### Score new loans
```bash
cd Kernels/Analytics
python score_loans.py \
  --input ../../Datasets/credit_test.csv \
  --output model_outputs/scored.csv \
  --threshold 0.27
```

Optional flags:
- `--threshold` — override decision cutoff  
- `--model` — path to a specific `.pkl`  

The scorer applies the same cleaning and feature engineering, aligns to the Standard-25 column list, and writes:

| Column | Meaning |
|--------|---------|
| `Default_Probability` | Model score in \[0, 1\] |
| `Predicted_Default` | 1 if probability ≥ threshold |
| `Threshold_Used` | Cutoff applied |

---

## 6. Key conclusions

1. **Default is predictable at moderate strength** — validation ROC-AUC ≈ **0.73** is useful for ranking and screening, not for perfect auto-decline.  
2. **Feature engineering drives signal** — ratios (LTI, DTI, utilisation), term, score bands, and **missing loan amount** rank among the strongest predictors.  
3. **A compact 25-feature set is enough** — little AUC gain from keeping all engineered columns after redundancy filters.  
4. **Threshold matters more than micro-tuning trees** — moving from 0.5 → ~0.27 (or cost-based 0.17) changes recall/precision far more than HPO did here.  
5. **Segment behaviour differs** — long-term loans and weak scores are easier to flag; strong scores with other stress factors produce more false negatives.  
6. **Operational recommendation**  
   - Use scores for **prioritisation and review queues**  
   - Default screening threshold **0.27** unless loss costs justify a lower cut  
   - Keep missingness flags at inference; do not drop incomplete applications silently  
   - Monitor PSI on top features and realised vs predicted default rates over time  

---

## 7. Main artifacts

| Path | Description |
|------|-------------|
| `features/credit_train_features.csv` | Engineered training matrix |
| `features/selected_features_standard.json` | Canonical 25-feature list |
| `features/feature_selection_ranking.csv` | Full consensus ranking |
| `model_outputs/best_model_calibrated.pkl` | Preferred production model |
| `model_outputs/best_model_hparam_gbm_default.pkl` | Uncalibrated GBM champion |
| `model_outputs/hparam_search_summary.json` | Search results + params |
| `model_outputs/calibration_threshold_meta.json` | Calibration + thresholds |
| `model_outputs/error_segment_analysis.csv` | FN/FP by segment |
| `model_outputs/scored_test_production.csv` | Example scored test file |

---

## 8. Limitations

- No application timestamps → limited drift / vintage analysis  
- No explicit loss-given-default or fee costs in the data extract  
- Protected attributes are not in the file; proxy fairness still needs policy review  
- Hyperparameter search used a subsample for speed; a larger grid may find small gains  
- Test labels are unavailable in this package — reported metrics are validation-based  

---

## 9. License & contact

Internal analytics package. Adapt thresholds and monitoring to your credit policy before production use.
