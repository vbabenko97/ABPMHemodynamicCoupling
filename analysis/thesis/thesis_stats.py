"""Reusable helper functions for thesis analysis notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def rank_biserial(
    x: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float] | None = None,
    paired: bool = False,
) -> tuple[float, float, float]:
    """Rank-biserial correlation as effect size for Mann-Whitney U or Wilcoxon signed-rank.

    Returns:
        (r, statistic, p_value)
    """
    x = np.asarray(x, dtype=float)

    if paired or y is None:
        if y is not None:
            diff = np.asarray(y, dtype=float) - x
        else:
            diff = x
        diff = diff[diff != 0]
        n = len(diff)
        result = stats.wilcoxon(diff)
        w_stat: float = float(result.statistic)
        p: float = float(result.pvalue)
        # Simple rank-biserial: r = 4*W+ / (n*(n+1)) - 1
        # Matches the bootstrap CI formula used in notebook 04.
        ranks = stats.rankdata(np.abs(diff))
        w_plus = float(np.sum(ranks[diff > 0]))
        r = 4 * w_plus / (n * (n + 1)) - 1 if n > 0 else 0.0
        return float(r), w_stat, p

    y_arr = np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y_arr)
    result = stats.mannwhitneyu(x, y_arr, alternative="two-sided")
    u_stat: float = float(result.statistic)
    p = float(result.pvalue)
    r = 1 - (2 * u_stat) / (n1 * n2)
    return float(r), u_stat, p


def bootstrap_ci(
    data: np.ndarray | Sequence[float],
    statistic_func: Callable[..., float] = np.median,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for any statistic function.

    Returns:
        (ci_low, ci_high, point_estimate)
    """
    data = np.asarray(data, dtype=float)
    point_estimate = float(statistic_func(data))
    rng = np.random.default_rng(seed)
    n = len(data)

    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = statistic_func(sample)

    alpha = 1 - ci
    ci_low = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return ci_low, ci_high, point_estimate


def bootstrap_corr_ci(
    x: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
    method: str = "spearman",
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Bootstrap CI for a correlation coefficient.

    Returns:
        (rho, ci_low, ci_high, p_value)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if method == "spearman":
        corr_func = stats.spearmanr
    elif method == "pearson":
        corr_func = stats.pearsonr
    elif method == "kendall":
        corr_func = stats.kendalltau
    else:
        msg = f"Unknown method: {method}"
        raise ValueError(msg)

    result = corr_func(x, y)
    rho = float(result.statistic) if hasattr(result, "statistic") else float(result[0])
    p_value = float(result.pvalue) if hasattr(result, "pvalue") else float(result[1])

    rng = np.random.default_rng(seed)
    n = len(x)
    boot_rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        res = corr_func(x[idx], y[idx])
        boot_rhos[i] = float(res.statistic) if hasattr(res, "statistic") else float(res[0])

    alpha = 1 - ci
    ci_low = float(np.percentile(boot_rhos, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_rhos, 100 * (1 - alpha / 2)))
    return rho, ci_low, ci_high, p_value


def derive_dipper_category(dip_pct: float | pd.Series) -> Any:
    """Classify nocturnal BP dipping percentage into clinical categories.

    Categories: Reverse dipper (<0%), Non-dipper (0-<10%),
    Normal dipper (10-<20%), Extreme dipper (>=20%).
    """
    if isinstance(dip_pct, pd.Series):
        return pd.cut(
            dip_pct,
            bins=[-np.inf, 0, 10, 20, np.inf],
            labels=["Reverse dipper", "Non-dipper", "Normal dipper", "Extreme dipper"],
            right=False,
        )
    if dip_pct < 0:
        return "Reverse dipper"
    if dip_pct < 10:
        return "Non-dipper"
    if dip_pct < 20:
        return "Normal dipper"
    return "Extreme dipper"


def _fmt_p(p: float) -> str:
    """Format p-value for APA style."""
    if np.isnan(p):
        return "p = NaN"
    if p < 0.001:
        return "p < .001"
    return f"p = .{int(round(p, 3) * 1000):03d}"


def format_apa_wilcoxon(
    W: float, p: float, r_rb: float, ci_low: float, ci_high: float, n: int
) -> str:
    """Format Wilcoxon result as APA string."""
    return (
        f"W = {W:.1f}, {_fmt_p(p)}, "
        f"r_rb = {r_rb:.2f}, 95% CI [{ci_low:.2f}, {ci_high:.2f}], n = {n}"
    )


def format_apa_mannwhitney(
    U: float, p: float, r_rb: float, ci_low: float, ci_high: float, n1: int, n2: int
) -> str:
    """Format Mann-Whitney result as APA string."""
    return (
        f"U = {U:.1f}, {_fmt_p(p)}, "
        f"r_rb = {r_rb:.2f}, 95% CI [{ci_low:.2f}, {ci_high:.2f}], "
        f"n1 = {n1}, n2 = {n2}"
    )


def format_apa_spearman(rs: float, p: float, ci_low: float, ci_high: float, n: int) -> str:
    """Format Spearman result as APA string."""
    return f"rs = {rs:.2f}, {_fmt_p(p)}, 95% CI [{ci_low:.2f}, {ci_high:.2f}], n = {n}"


def format_apa_friedman(chi2: float, p: float, W: float, df: int, n: int) -> str:
    """Format Friedman result as APA string."""
    return f"\u03c7\u00b2({df}) = {chi2:.2f}, {_fmt_p(p)}, W = {W:.2f}, n = {n}"


def format_apa_kruskal(H: float, p: float, epsilon_sq: float, df: int, n: int) -> str:
    """Format Kruskal-Wallis result as APA string."""
    return f"H({df}) = {H:.2f}, {_fmt_p(p)}, \u03b5\u00b2 = {epsilon_sq:.3f}, n = {n}"


def format_median_iqr(data: np.ndarray | Sequence[float]) -> str:
    """Format as 'median (Q1-Q3)' string."""
    arr = np.asarray(data, dtype=float)
    median = np.median(arr)
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    return f"{median:.1f} ({q1:.1f}\u2013{q3:.1f})"


def format_mean_sd(data: np.ndarray | Sequence[float]) -> str:
    """Format as 'mean +/- SD' string."""
    arr = np.asarray(data, dtype=float)
    return f"{np.mean(arr):.1f} \u00b1 {np.std(arr, ddof=1):.1f}"


def compute_icc1(
    df_long: pd.DataFrame,
    subject_col: str,
    measure_col: str,
) -> tuple[float, float | None, float | None]:
    """Compute ICC(1) using one-way random ANOVA decomposition.

    Returns:
        (icc, ci_low, ci_high) — CI is None when computed manually.
    """
    try:
        import pingouin as pg  # type: ignore[import-untyped]

        # pingouin expects a 'rater' column; create one from within-subject ordering
        df_work = df_long.copy()
        df_work["_rater"] = df_work.groupby(subject_col).cumcount()
        icc_df = pg.intraclass_corr(
            data=df_work,
            targets=subject_col,
            raters="_rater",
            ratings=measure_col,
        )
        row = icc_df[icc_df["Type"] == "ICC1"].iloc[0]
        return float(row["ICC"]), float(row["CI95%"][0]), float(row["CI95%"][1])
    except ImportError:
        pass

    # Manual one-way random ANOVA decomposition
    groups = [g[measure_col].values for _, g in df_long.groupby(subject_col)]
    k_values = np.array([len(g) for g in groups])
    n_subjects = len(groups)
    grand_mean = df_long[measure_col].mean()

    # Between-subject sum of squares
    group_means = np.array([g.mean() for g in groups])
    ss_between = float(np.sum(k_values * (group_means - grand_mean) ** 2))
    df_between = n_subjects - 1

    # Within-subject sum of squares
    ss_within = float(sum(np.sum((g - g.mean()) ** 2) for g in groups))
    df_within = int(np.sum(k_values) - n_subjects)

    ms_between = ss_between / df_between if df_between > 0 else 0.0
    ms_within = ss_within / df_within if df_within > 0 else 0.0

    k0 = float(np.mean(k_values))
    icc = (ms_between - ms_within) / (ms_between + (k0 - 1) * ms_within) if ms_between > 0 else 0.0
    return float(icc), None, None


def variance_decomposition(
    df_long: pd.DataFrame,
    subject_col: str,
    measure_col: str,
) -> dict[str, float]:
    """Decompose total variance into between-subject and within-subject components.

    Returns:
        {"between_var": float, "within_var": float, "total_var": float, "icc1": float}
    """
    groups = [g[measure_col].values for _, g in df_long.groupby(subject_col)]
    k_values = np.array([len(g) for g in groups])
    n_subjects = len(groups)
    grand_mean = df_long[measure_col].mean()

    group_means = np.array([g.mean() for g in groups])
    ss_between = float(np.sum(k_values * (group_means - grand_mean) ** 2))
    df_between = n_subjects - 1

    ss_within = float(sum(np.sum((g - g.mean()) ** 2) for g in groups))
    df_within = int(np.sum(k_values) - n_subjects)

    ms_between = ss_between / df_between if df_between > 0 else 0.0
    ms_within = ss_within / df_within if df_within > 0 else 0.0

    k0 = float(np.mean(k_values))
    between_var = (ms_between - ms_within) / k0 if ms_between > ms_within else 0.0
    within_var = ms_within
    total_var = between_var + within_var
    icc1 = between_var / total_var if total_var > 0 else 0.0

    return {
        "between_var": between_var,
        "within_var": within_var,
        "total_var": total_var,
        "icc1": icc1,
    }


def export_table(
    df: pd.DataFrame,
    path: str | Path,
    formats: tuple[str, ...] = ("csv", "tex"),
) -> None:
    """Export a DataFrame to CSV and/or LaTeX format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    stem = path.stem
    parent = path.parent

    if "csv" in formats:
        df.to_csv(parent / f"{stem}.csv", index=False)
    if "tex" in formats:
        df.to_latex(parent / f"{stem}.tex", escape=False, index=False)


def add_sin_cos_hour(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
    """Add sin_hour and cos_hour columns for circadian modeling."""
    hour = (
        pd.to_datetime(df[datetime_col]).dt.hour + pd.to_datetime(df[datetime_col]).dt.minute / 60
    )
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24)
    return df
