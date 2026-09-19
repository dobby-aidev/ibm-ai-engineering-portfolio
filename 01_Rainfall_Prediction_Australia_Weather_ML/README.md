# Rainfall Prediction in Australia (Weather AUS)

Final project for the **Machine Learning with Python** course within the **IBM AI Engineering Professional Certificate** on Coursera.

- **Verified Certificate:** [Coursera Accomplishment (Verify Credential)](https://coursera.org/share/deb792b19739e807ee06ccc96fded4c3)
- **Frameworks:** Python, Scikit-Learn, Pandas, NumPy, Seaborn, Matplotlib

---

## Overview

This project builds an end-to-end classification pipeline to predict whether it will rain tomorrow in Australia based on 10+ years of daily weather observations from the Australian Government's Bureau of Meteorology.

The challenge involves handling class imbalance, missing observation values, mixed numerical/categorical feature sets, and tuning models with cross-validation without data leakage.

---

## Dataset

- **Source:** Australian Government Bureau of Meteorology (`weatherAUS-2.csv`)
- **Target Variable:** `RainTomorrow` (Binary: `Yes` / `No`)
- **Key Features:**
  - Atmospheric: `MinTemp`, `MaxTemp`, `Rainfall`, `Evaporation`, `Sunshine`
  - Wind: `WindGustDir`, `WindGustSpeed`, `WindDir9am`, `WindDir3pm`, `WindSpeed9am`, `WindSpeed3pm`
  - Moisture & Pressure: `Humidity9am`, `Humidity3pm`, `Pressure9am`, `Pressure3pm`
  - Cloud cover: `Cloud9am`, `Cloud3pm`
  - Temperature: `Temp9am`, `Temp3pm`

---

## Pipeline Architecture

To avoid data leakage, all preprocessing transformations are encapsulated within a unified Scikit-Learn `Pipeline`:

1. **Preprocessing (`ColumnTransformer`):**
   - **Numerical Features:** Imputed and standardized using `StandardScaler`.
   - **Categorical Features:** Encoded using `OneHotEncoder(handle_unknown='ignore')`.
2. **Cross-Validation Strategy:**
   - 5-Fold `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` ensuring consistent class distributions across folds.
3. **Hyperparameter Optimization (`GridSearchCV`):**
   - **Model 1: Random Forest Classifier**
     - Hyperparameters tuned: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`.
   - **Model 2: Logistic Regression**
     - Hyperparameters tuned: `penalty` (`l1`, `l2`), `solver='liblinear'`, `class_weight` (`None`, `'balanced'`).

---

## Evaluation & Results

Evaluated on an independent holdout test set (1,512 observations, 80/20 train/test split):

| Model | Test Accuracy | Precision (Rain: Yes) | Recall (Rain: Yes) | Macro F1 |
|---|---|---|---|---|
| **Random Forest (Tuned)** | **84%** | 0.75 | 0.51 | 0.76 |
| **Logistic Regression (Tuned)** | **83%** | 0.69 | 0.51 | 0.74 |

- **Feature Importance:** Extracted from the optimal Random Forest estimator, identifying `Humidity3pm`, `Sunshine`, `Pressure3pm`, and `WindGustSpeed` as the strongest predictive signals for next-day precipitation.
- **Diagnostics:** Confusion matrix heatmaps plotted to assess false positive vs false negative trade-offs.

---

## Project Structure

```text
01_Rainfall_Prediction_Australia_Weather_ML/
├── FinalProject_AUSWeather.ipynb   # Fully executed project notebook with outputs
└── README.md                       # Project documentation
```

---

## Running the Code

1. Install requirements:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```
2. Launch notebook:
   ```bash
   jupyter notebook FinalProject_AUSWeather.ipynb
   ```
