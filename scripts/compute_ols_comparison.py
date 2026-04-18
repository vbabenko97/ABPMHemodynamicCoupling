"""Compare Brandon feature selection vs OLS with individual/all nonlinear features.

Addresses reviewer concern: is Brandon's advantage due to algorithm or feature pool?
Uses the same 3-fold temporal block CV as the main pipeline.
"""

import sys
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = "/Users/vitaliibabenko/babenko-dev/ABPMHemodynamicCoupling"
sys.path.insert(0, PROJECT_ROOT)

from src.config import Columns, Config
from src.data_processing import DataLoader
from src.feature_engineering import DBPFeatureExtractor
from src.modeling import BrandonSelector

FEATURE_NAMES = ["SBP", "HR", "1/SBP", "1/HR", "SBP×HR", "1/(SBP×HR)"]


def cv_mae_for_features(X_raw, y, feature_indices, config):
    """3-fold temporal block CV MAE for OLS on given feature subset."""
    kf = KFold(n_splits=config.N_CV_SPLITS, shuffle=False)
    maes = []
    for train_ix, val_ix in kf.split(X_raw):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_raw[train_ix])
        X_val = scaler.transform(X_raw[val_ix])
        try:
            model = LinearRegression().fit(X_train[:, feature_indices], y[train_ix])
            pred = model.predict(X_val[:, feature_indices])
            maes.append(mean_absolute_error(y[val_ix], pred))
        except (np.linalg.LinAlgError, ValueError):
            maes.append(np.nan)
    valid = [m for m in maes if not np.isnan(m)]
    return np.mean(valid) if valid else np.inf


def cv_mae_brandon(X_raw, y, config):
    """3-fold temporal block CV MAE for Brandon selector."""
    kf = KFold(n_splits=config.N_CV_SPLITS, shuffle=False)
    maes = []
    for train_ix, val_ix in kf.split(X_raw):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_raw[train_ix])
        X_val = scaler.transform(X_raw[val_ix])
        try:
            selector = BrandonSelector(config)
            model, idx = selector.fit(X_train, y[train_ix])
            pred = model.predict(X_val[:, idx])
            maes.append(mean_absolute_error(y[val_ix], pred))
        except (np.linalg.LinAlgError, ValueError):
            maes.append(np.nan)
    valid = [m for m in maes if not np.isnan(m)]
    return np.mean(valid) if valid else np.inf


def cv_mae_lasso(X_raw, y, config):
    """3-fold temporal block CV MAE for LassoCV on all features."""
    from sklearn.exceptions import ConvergenceWarning
    kf = KFold(n_splits=config.N_CV_SPLITS, shuffle=False)
    maes = []
    for train_ix, val_ix in kf.split(X_raw):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_raw[train_ix])
        X_val = scaler.transform(X_raw[val_ix])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model = LassoCV(
                    cv=config.LASSO_CV_FOLDS,
                    max_iter=config.LASSO_MAX_ITER,
                    random_state=config.RANDOM_SEED,
                    selection=config.LASSO_SELECTION,
                    tol=config.LASSO_TOL,
                ).fit(X_train, y[train_ix])
            pred = model.predict(X_val)
            maes.append(mean_absolute_error(y[val_ix], pred))
        except (Exception,):
            maes.append(np.nan)
    valid = [m for m in maes if not np.isnan(m)]
    return np.mean(valid) if valid else np.inf


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    config = Config()
    np.random.seed(config.RANDOM_SEED)
    loader = DataLoader(config)
    df = loader.load_monitoring_data()
    extractor = DBPFeatureExtractor(config)

    # Models to compare: all 6 individual OLS + OLS(All6) + OLS(SBP,HR) + Brandon + Lasso
    single_models = {f"OLS({FEATURE_NAMES[i]})": [i] for i in range(6)}
    combo_models = {
        "OLS(SBP,HR)": [0, 1],
        "OLS(All6)": list(range(6)),
    }
    ols_models = {**single_models, **combo_models}

    rows = []
    subjects = sorted(df[Columns.PAT_ID].unique())

    for sid in subjects:
        df_subj = df[df[Columns.PAT_ID] == sid]
        df_base = df_subj[df_subj[Columns.LABEL] == Columns.LABEL_BASELINE]
        if len(df_base) < config.MIN_BASELINE_SAMPLES:
            continue
        X_raw = extractor.extract(df_base)
        y = df_base[Columns.DBP].values
        if len(y) < config.N_CV_SPLITS * config.MIN_CV_SAMPLES_MULTIPLIER:
            continue

        row = {"subject": sid}

        # OLS variants
        for name, idxs in ols_models.items():
            row[name] = cv_mae_for_features(X_raw, y, idxs, config)

        # Brandon
        row["Brandon"] = cv_mae_brandon(X_raw, y, config)

        # Lasso (all 6 features, regularized)
        row["Lasso"] = cv_mae_lasso(X_raw, y, config)

        rows.append(row)
        print(f"  Subject {sid}: done")

    results = pd.DataFrame(rows)
    n = len(results)
    print(f"\n{'='*60}")
    print(f"COMPARISON RESULTS (N={n})")
    print(f"{'='*60}\n")

    # Summary statistics
    model_cols = [c for c in results.columns if c != "subject"]
    print("Median CV-MAE [IQR] per model:")
    for col in model_cols:
        vals = results[col].dropna()
        med = np.median(vals)
        q1, q3 = np.percentile(vals, [25, 75])
        print(f"  {col:20s}: {med:.2f} [{q1:.2f}, {q3:.2f}] mmHg")

    # Pairwise comparisons: Brandon vs each
    print(f"\n{'='*60}")
    print("PAIRWISE: Brandon vs each model (Wilcoxon signed-rank)")
    print(f"{'='*60}")
    brandon = results["Brandon"]
    for col in model_cols:
        if col == "Brandon":
            continue
        other = results[col]
        diff = brandon - other
        valid = diff.dropna()
        if len(valid) < 5:
            continue
        med_diff = np.median(valid)
        brandon_wins = (valid < 0).sum()
        other_wins = (valid > 0).sum()
        try:
            stat, p = stats.wilcoxon(valid)
        except ValueError:
            stat, p = np.nan, np.nan
        print(f"  Brandon vs {col:20s}: Δ={med_diff:+.2f} mmHg, "
              f"Brandon wins {brandon_wins}/{len(valid)}, W={stat:.0f}, p={p:.4f}")

    # Winner counts
    print(f"\n{'='*60}")
    print("WINNER COUNTS (best CV-MAE per subject)")
    print(f"{'='*60}")
    winner_col = results[model_cols].idxmin(axis=1)
    counts = winner_col.value_counts()
    for name, cnt in counts.items():
        print(f"  {name:20s}: {cnt} ({100*cnt/n:.1f}%)")

    # Key comparison: OLS(All6) vs Brandon
    print(f"\n{'='*60}")
    print("KEY: OLS(All6) vs Brandon")
    print(f"{'='*60}")
    diff = results["Brandon"] - results["OLS(All6)"]
    med_diff = np.median(diff)
    brandon_better = (diff < 0).sum()
    ols_all_better = (diff > 0).sum()
    stat, p = stats.wilcoxon(diff)
    print(f"  Median Δ(Brandon - OLS(All6)) = {med_diff:+.2f} mmHg")
    print(f"  Brandon better: {brandon_better}/{n}, OLS(All6) better: {ols_all_better}/{n}")
    print(f"  Wilcoxon W={stat:.0f}, p={p:.4f}")

    # Save detailed results
    out_path = f"{PROJECT_ROOT}/docs/thesis/tables/ols_comparison_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nDetailed per-subject results saved to: {out_path}")


if __name__ == "__main__":
    main()
