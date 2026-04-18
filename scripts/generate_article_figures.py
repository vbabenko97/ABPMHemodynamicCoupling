"""Generate publication-quality figures for a Ukrainian scientific article."""

import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = "/Users/vitaliibabenko/babenko-dev/ABPMHemodynamicCoupling"
sys.path.insert(0, PROJECT_ROOT)

from src.config import Columns, Config  # noqa: E402
from src.data_processing import DataLoader  # noqa: E402
from src.feature_engineering import DBPFeatureExtractor  # noqa: E402
from src.modeling import CrossValidator, ModelTrainer  # noqa: E402

OUTPUT_DIR = f"{PROJECT_ROOT}/docs/thesis/figures"

# ---------------------------------------------------------------------------
# Global matplotlib style
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ---------------------------------------------------------------------------
# Figure 1 -- Predictor frequency bar chart
# ---------------------------------------------------------------------------
def generate_figure1() -> str:
    print("Generating Figure 1: predictor frequency bar chart...")

    data = [
        ("САТ", 15),
        ("1/САТ", 15),
        ("ЧСС", 10),
        ("САТ\u00d7ЧСС", 9),
        ("1/(САТ\u00d7ЧСС)", 9),
        ("1/ЧСС", 7),
    ]
    labels = [d[0] for d in data]
    counts = [d[1] for d in data]
    total = 28

    # Sort ascending so highest ends up at top in barh
    order = np.argsort(counts)
    labels_sorted = [labels[i] for i in order]
    counts_sorted = [counts[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels_sorted, counts_sorted, color="steelblue", edgecolor="white")

    for bar, count in zip(bars, counts_sorted, strict=True):
        pct = 100 * count / total
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.0f}%",
            va="center",
            fontsize=10,
        )

    ax.set_xlabel("Кількість учасників")
    ax.set_title("")
    ax.set_xlim(0, max(counts_sorted) + 3)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig1_predictor_frequency.png"
    fig.savefig(path, dpi=400)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 2 -- Condition boxplots (MAE inflation + delta-bias)
# ---------------------------------------------------------------------------
def compute_condition_metrics() -> pd.DataFrame:
    print("Computing per-subject condition metrics...")
    warnings.filterwarnings("ignore", category=FutureWarning)

    config = Config()
    np.random.seed(config.RANDOM_SEED)
    loader = DataLoader(config)
    df = loader.load_monitoring_data()

    cv_validator = CrossValidator(config)
    trainer = ModelTrainer(config)
    dbp_extractor = DBPFeatureExtractor(config)

    conditions = [
        ("Когнітивне", Columns.LABEL_COGNITIVE_TASK),
        ("Фізичне", Columns.LABEL_PHYSICAL_TASK),
        ("Повітряна тривога", Columns.LABEL_AIR_ALERT),
    ]

    rows: list[dict[str, object]] = []
    for subject_id in df[Columns.PAT_ID].unique():
        df_subj = df[df[Columns.PAT_ID] == subject_id]
        df_base = df_subj[df_subj[Columns.LABEL] == Columns.LABEL_BASELINE]

        if len(df_base) < config.MIN_BASELINE_SAMPLES:
            continue

        X_raw = dbp_extractor.extract(df_base)
        y = df_base[Columns.DBP].values

        cv_perf = cv_validator.evaluate_models(X_raw, y)
        if cv_perf is None:
            continue

        winner = cv_perf.get_winner()
        ref_mae = cv_perf.get_best_score()

        model, idxs, scaler = trainer.train(X_raw, y, winner)

        for cond_name, label in conditions:
            df_cond = df_subj[df_subj[Columns.LABEL] == label]
            if df_cond.empty:
                continue

            X_cond = dbp_extractor.extract(df_cond)
            X_cond_scaled = scaler.transform(X_cond)
            y_true = df_cond[Columns.DBP].values
            y_pred = model.predict(X_cond_scaled[:, idxs])

            mae = float(np.mean(np.abs(y_true - y_pred)))
            bias = float(np.median(y_true - y_pred))
            inflation = 100.0 * (mae - ref_mae) / (ref_mae + config.EPSILON)

            rows.append(
                {
                    "participant_id": subject_id,
                    "condition": cond_name,
                    "mae_inflation_pct": inflation,
                    "delta_bias_mmhg": bias,
                }
            )

    return pd.DataFrame(rows)


def generate_figure2(metrics_df: pd.DataFrame) -> str:
    print("Generating Figure 2: condition boxplots...")

    condition_order = ["Когнітивне", "Фізичне", "Повітряна тривога"]
    palette = {"Когнітивне": "steelblue", "Фізичне": "coral", "Повітряна тривога": "mediumseagreen"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    for ax, col, ylabel in [
        (ax1, "mae_inflation_pct", "Зростання MAE, %"),
        (ax2, "delta_bias_mmhg", "Зсув залишків, мм рт. ст."),
    ]:
        bp_data = [
            metrics_df.loc[metrics_df["condition"] == c, col].dropna().values
            for c in condition_order
        ]
        bp = ax.boxplot(
            bp_data,
            positions=range(len(condition_order)),
            widths=0.5,
            patch_artist=True,
            showfliers=False,
        )
        for patch, cond in zip(bp["boxes"], condition_order, strict=True):
            patch.set_facecolor(palette[cond])
            patch.set_alpha(0.6)
        for element in ("whiskers", "caps", "medians"):
            for line in bp[element]:
                line.set_color("black")

        # Jittered individual points
        rng = np.random.default_rng(42)
        for i, cond in enumerate(condition_order):
            vals = metrics_df.loc[metrics_df["condition"] == cond, col].dropna().values
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(
                i + jitter,
                vals,
                color=palette[cond],
                edgecolor="white",
                s=25,
                alpha=0.8,
                zorder=3,
            )

        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xticks(range(len(condition_order)))
        ax.set_xticklabels(condition_order, fontsize=10)
        ax.set_ylabel(ylabel)

    fig.suptitle("")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig1_condition_boxplots.png"
    fig.savefig(path, dpi=400)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    path1 = generate_figure1()
    metrics = compute_condition_metrics()
    path2 = generate_figure2(metrics)

    print("\nDone. Saved figures:")
    print(f"  {path1}")
    print(f"  {path2}")
