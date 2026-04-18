"""
Streamlit page for the synthetic stress/workload simulator.
"""

from __future__ import annotations

import time
from dataclasses import asdict

import pandas as pd
import streamlit as st

from src.simulator import PersonProfile, PhysiologySimulator

STATUS_LABELS = {
    "warmup": "Розігрів і побудова baseline",
    "baseline": "Базовий стан",
    "elevated": "Підвищена фізіологічна відповідь",
    "stress_response": "Stress response",
}

SEX_OPTIONS = {
    "Чоловік": "male",
    "Жінка": "female",
    "Інше": "other",
}

EMPTY_HISTORY_COLUMNS = [
    "t",
    "HR",
    "SBP",
    "DBP",
    "HR_baseline",
    "SBP_baseline",
    "DBP_baseline",
    "HR_upper_band",
    "SBP_upper_band",
    "DBP_upper_band",
    "z_hr",
    "z_sbp",
    "z_dbp",
    "response_score",
    "status",
    "active_events",
    "arousal_state",
    "workload_state",
    "physical_state",
    "fatigue_state",
]


def build_profile_from_sidebar() -> PersonProfile:
    """Read simulator profile inputs from the sidebar."""
    st.sidebar.header("Профіль оператора")
    age = st.sidebar.slider("Вік", 18, 75, 32)
    sex_label = st.sidebar.selectbox("Стать", list(SEX_OPTIONS.keys()), index=0)
    fitness = st.sidebar.slider("Фізична форма", 1, 5, 3, help="1 = низька, 5 = спортивна")
    trait_anxiety = st.sidebar.slider("Тривожність як риса", 0.0, 1.0, 0.35, 0.05)
    caffeine_cups = st.sidebar.slider("Кава на день", 0, 6, 1)
    smoker = st.sidebar.checkbox("Куріння", value=False)
    on_beta_blocker = st.sidebar.checkbox("Бета-блокатор", value=False)

    return PersonProfile(
        age=age,
        sex=SEX_OPTIONS[sex_label],
        fitness=fitness,
        trait_anxiety=trait_anxiety,
        caffeine_cups=caffeine_cups,
        smoker=smoker,
        on_beta_blocker=on_beta_blocker,
    )


def build_settings_from_sidebar() -> dict[str, int | float | bool | str]:
    """Read simulator settings from the sidebar."""
    st.sidebar.header("Симуляція")
    warmup = st.sidebar.slider("Warmup, с", 20, 180, 60, 10)
    random_events = st.sidebar.checkbox("Увімкнути випадкові події", value=True)
    event_rate = st.sidebar.slider("Подій за хвилину", 0.0, 2.0, 0.4, 0.1)
    tail_seconds = st.sidebar.slider("Видима історія, с", 60, 600, 180, 30)
    seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)
    focus_metric = st.sidebar.selectbox("Канал baseline-графіка", ["HR", "SBP", "DBP"], index=0)

    return {
        "warmup": int(warmup),
        "random_events": bool(random_events),
        "event_rate": float(event_rate),
        "tail_seconds": int(tail_seconds),
        "seed": int(seed),
        "focus_metric": focus_metric,
    }


def create_simulator(
    profile: PersonProfile,
    settings: dict[str, int | float | bool | str],
) -> PhysiologySimulator:
    """Build a new simulator instance from the current controls."""
    return PhysiologySimulator(
        profile=profile,
        warmup_seconds=int(settings["warmup"]),
        seed=int(settings["seed"]),
        random_events_enabled=bool(settings["random_events"]),
        event_rate_per_min=float(settings["event_rate"]),
    )


def ensure_simulator_state(
    profile: PersonProfile,
    settings: dict[str, int | float | bool | str],
) -> None:
    """Initialize session state for the simulator page."""
    if "simulator_running" not in st.session_state:
        st.session_state.simulator_running = False
    if "simulator_history" not in st.session_state:
        st.session_state.simulator_history = []
    if "simulator_engine" not in st.session_state:
        st.session_state.simulator_engine = create_simulator(profile, settings)
        st.session_state.simulator_profile_snapshot = asdict(profile)
        st.session_state.simulator_settings_snapshot = settings.copy()


def reset_simulator(
    profile: PersonProfile,
    settings: dict[str, int | float | bool | str],
) -> None:
    """Reset the simulator and clear all accumulated history."""
    st.session_state.simulator_running = False
    st.session_state.simulator_history = []
    st.session_state.simulator_engine = create_simulator(profile, settings)
    st.session_state.simulator_profile_snapshot = asdict(profile)
    st.session_state.simulator_settings_snapshot = settings.copy()


def append_simulation_step() -> None:
    """Advance the simulator by one second and store the new row."""
    row = st.session_state.simulator_engine.step()
    st.session_state.simulator_history.append(row)


def history_frame() -> pd.DataFrame:
    """Return history as a dataframe with stable columns."""
    if not st.session_state.simulator_history:
        return pd.DataFrame(columns=EMPTY_HISTORY_COLUMNS)
    return pd.DataFrame(st.session_state.simulator_history)


st.set_page_config(
    page_title="Синтетичний симулятор stress/workload",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

st.title("Синтетичний симулятор stress/workload")
st.caption(
    "Окрема demo-сторінка для UI та detector prototyping. "
    "Це не клінічна модель і не замінює реальні ABPM-дані."
)

profile = build_profile_from_sidebar()
settings = build_settings_from_sidebar()
ensure_simulator_state(profile, settings)

if (
    st.session_state.simulator_profile_snapshot != asdict(profile)
    or st.session_state.simulator_settings_snapshot != settings
):
    st.info("Параметри профілю або симуляції змінилися. Натисніть `Reset`, щоб застосувати їх чисто.")

control_start, control_stop, control_step, control_reset = st.columns(4)
with control_start:
    if st.button("Start", use_container_width=True):
        st.session_state.simulator_running = True
with control_stop:
    if st.button("Stop", use_container_width=True):
        st.session_state.simulator_running = False
with control_step:
    if st.button("Крок +1 с", use_container_width=True):
        append_simulation_step()
        st.rerun()
with control_reset:
    if st.button("Reset", use_container_width=True):
        reset_simulator(profile, settings)
        st.rerun()

st.subheader("Ручні події")
manual_events = [
    ("alarm", "Сигнал"),
    ("deadline", "Дедлайн"),
    ("interruption", "Переривання"),
    ("social", "Соціальний тиск"),
    ("physical", "Фізичний рух"),
    ("recovery", "Відновлення"),
]
for column, (key, label) in zip(st.columns(len(manual_events)), manual_events, strict=False):
    with column:
        if st.button(label, use_container_width=True):
            st.session_state.simulator_engine.spawn_event(key)
            st.rerun()

if st.session_state.simulator_running:
    append_simulation_step()

history_df = history_frame()
tail_df = history_df.tail(int(settings["tail_seconds"])).copy()

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
if tail_df.empty:
    metric_col1.metric("HR", "-")
    metric_col2.metric("SBP / DBP", "-")
    metric_col3.metric("Response score", "-")
    metric_col4.metric("Статус", "Очікування")
    latest = None
else:
    latest = tail_df.iloc[-1]
    metric_col1.metric("HR", f"{latest['HR']:.1f} bpm")
    metric_col2.metric("SBP / DBP", f"{latest['SBP']:.1f} / {latest['DBP']:.1f} мм рт. ст.")
    metric_col3.metric("Response score", f"{latest['response_score']:.2f}")
    metric_col4.metric("Статус", STATUS_LABELS.get(str(latest["status"]), str(latest["status"])))

left_chart, right_chart = st.columns(2)
with left_chart:
    st.markdown("**Синтетичні вітальні сигнали**")
    if tail_df.empty:
        st.write("Ще немає вимірювань. Натисніть `Start` або `Крок +1 с`.")
    else:
        st.line_chart(tail_df.set_index("t")[["HR", "SBP", "DBP"]])
        active_text = str(latest["active_events"]).strip() if latest is not None else ""
        st.caption(f"Активні події: {active_text or 'немає'}")

with right_chart:
    st.markdown("**Персоналізований baseline**")
    focus_metric = str(settings["focus_metric"])
    baseline_col = f"{focus_metric}_baseline"
    if tail_df.empty or tail_df[baseline_col].isna().all():
        st.write(
            f"Baseline ще формується. Потрібно приблизно {settings['warmup']} секунд даних."
        )
    else:
        focus_plot = tail_df.set_index("t")[
            [focus_metric, baseline_col, f"{focus_metric}_upper_band"]
        ]
        st.line_chart(focus_plot)
        st.caption(
            f"Показано {focus_metric}, адаптивний baseline та верхню межу baseline + 2σ."
        )

left_state, right_state = st.columns(2)
with left_state:
    st.markdown("**Приховані стани симулятора**")
    if tail_df.empty:
        st.write("Траєкторія latent states з’явиться після перших кроків.")
    else:
        st.line_chart(
            tail_df.set_index("t")[
                ["arousal_state", "workload_state", "physical_state", "fatigue_state"]
            ]
        )

with right_state:
    st.markdown("**Response score і каналові відхилення**")
    if tail_df.empty:
        st.write("Поки що немає score.")
    else:
        st.line_chart(tail_df.set_index("t")[["response_score", "z_hr", "z_sbp", "z_dbp"]])

with st.expander("Лог подій та експорт", expanded=False):
    if st.session_state.simulator_engine.event_log:
        event_log_df = pd.DataFrame(st.session_state.simulator_engine.event_log).tail(20)
        st.dataframe(event_log_df, use_container_width=True, hide_index=True)
    else:
        st.write("Подій ще не було.")

    if not history_df.empty:
        st.download_button(
            label="Завантажити CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="stress_load_simulation.csv",
            mime="text/csv",
        )

st.divider()
baselines = st.session_state.simulator_engine.baselines
st.markdown(
    "**Поточні baseline-priors профілю**: "
    f"HR ~ {baselines['HR']:.1f} bpm, "
    f"SBP ~ {baselines['SBP']:.1f} мм рт. ст., "
    f"DBP ~ {baselines['DBP']:.1f} мм рт. ст."
)

if st.session_state.simulator_running:
    time.sleep(1.0)
    st.rerun()
