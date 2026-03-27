"""
ABPM Hemodynamic Coupling — Streamlit Web Interface
=====================================================

Upload ABPM monitoring data and run the analysis pipeline interactively.

Usage:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st

from src.web_pipeline import PipelineResults, WebPipeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ABPM Гемодинамічне Поєднання",
    page_icon=":anatomical_heart:",
    layout="wide",
)

st.title("Аналіз гемодинамічного поєднання ABPM")
st.markdown(
    "Завантажте дані амбулаторного моніторингу артеріального тиску, "
    "щоб запустити повний конвеєр аналізу на рівні окремих учасників "
    "і всієї когорти."
)

# ---------------------------------------------------------------------------
# Sidebar — file upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Завантаження даних")
    uploaded_file = st.file_uploader(
        "Завантажте `monitoring_data.csv`",
        type=["csv"],
        help="CSV-файл зі стовпцями: participant_id, datetime, SBP, DBP, HR, "
        "а також індикаторами контексту (state, alert_window, is_cog, is_phys, …)",
    )

    if uploaded_file is not None:
        st.success(f"**{uploaded_file.name}** ({uploaded_file.size:,} bytes)")
        run_clicked = st.button(
            "Запустити аналіз",
            type="primary",
            use_container_width=True,
        )
    else:
        run_clicked = False
        st.info("Очікується завантаження CSV-файлу.")

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
if run_clicked and uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    pipeline = WebPipeline()

    with st.status("Виконується аналіз…", expanded=True) as status:
        # Stage 1 — sanitize & validate
        st.write("Очищення та перевірка даних…")
        cleaned_df, san_report = pipeline.sanitize(df_raw)

        if san_report.has_drops:
            details = ", ".join(
                f"**{n}** рядків із некоректним значенням {col}"
                for col, n in san_report.counts.items()
            )
            st.warning(
                f"Видалено {san_report.total_dropped} рядків під час очищення: {details}"
            )

        try:
            df = pipeline.validate_and_preprocess(cleaned_df)
        except ValueError as exc:
            st.error(f"Перевірка даних не пройдена: {exc}")
            st.stop()

        n_subjects = df["participant_id"].nunique()
        n_records = len(df)
        st.write(
            f"Завантажено **{n_records:,}** записів для **{n_subjects}** учасників."
        )

        # Stage 2 — per-subject analysis
        st.write("Аналіз учасників…")
        progress = st.progress(0)
        res_df = pipeline.analyze_subjects(df, progress_callback=progress.progress)
        progress.empty()

        # Stage 3 — cohort statistics
        st.write("Обчислення статистики когорти…")
        summary_text = pipeline.compute_statistics(res_df)

        # Stage 4 — figures
        st.write("Побудова графіків…")
        demographics_fig = pipeline.create_demographics_figure(df)
        figures = pipeline.generate_figures(df, res_df)

        status.update(label="Аналіз завершено!", state="complete", expanded=False)

    st.session_state.results = PipelineResults(
        subject_metrics=res_df,
        demographics_figure=demographics_fig,
        summary_text=summary_text,
        figures=figures,
        n_subjects=n_subjects,
        n_records=n_records,
    )
    if san_report.has_drops:
        st.session_state.excluded_rows = san_report.dropped_rows
    else:
        st.session_state.pop("excluded_rows", None)

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
if "results" in st.session_state:
    results: PipelineResults = st.session_state.results

    st.divider()

    # Show excluded rows if any were dropped during sanitization
    if "excluded_rows" in st.session_state:
        excluded = st.session_state.excluded_rows
        with st.expander(f"Виключені рядки ({len(excluded)})", expanded=False):
            st.dataframe(excluded, use_container_width=True)
            st.download_button(
                "Завантажити виключені рядки",
                excluded.to_csv(index=False),
                file_name="excluded_rows.csv",
                mime="text/csv",
            )

    col1, col2, col3 = st.columns(3)
    col1.metric("Учасники", results.n_subjects)
    col2.metric("Записи", f"{results.n_records:,}")
    col3.metric("Проаналізовані умови", len(results.figures))

    tab_summary, tab_metrics, tab_figures = st.tabs(
        ["Підсумок", "Метрики учасників", "Графіки"]
    )

    # --- Summary tab ---
    with tab_summary:
        st.subheader("Демографія")
        st.pyplot(results.demographics_figure)

        st.subheader("Підсумок результатів")
        st.code(results.summary_text, language=None)

    # --- Subject metrics tab ---
    with tab_metrics:
        st.subheader("Метрики по кожному учаснику")
        st.dataframe(results.subject_metrics, use_container_width=True)

        st.download_button(
            "Завантажити CSV",
            results.subject_metrics.to_csv(index=False),
            file_name="per_subject_metrics.csv",
            mime="text/csv",
        )

    # --- Figures tab ---
    with tab_figures:
        for name, fig in results.figures.items():
            st.subheader(name.replace("_", " ").title())
            st.pyplot(fig)
