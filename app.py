"""
ABPM Hemodynamic Coupling — Streamlit Web Interface
=====================================================

Upload ABPM monitoring data and run the analysis pipeline interactively.

Usage:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st

from abpm_hemodynamic_coupling.web_pipeline import PipelineResults, WebPipeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Система оцінки тонусу периферичних артеріальних судин",
    page_icon=":anatomical_heart:",
    layout="wide",
)

st.title("Система оцінки тонусу периферичних артеріальних судин")
st.caption("на основі даних добового моніторингу артеріального тиску (ДМАТ)")

st.info(
    "**Як система оцінює тонус.** Діастолічний артеріальний тиск (ДАТ) "
    "формується переважно загальним периферичним судинним опором. Тому "
    "стійкий зв'язок ДАТ ↔ (САТ, ЧСС) у межах доби є функціональним проявом "
    "тонусу периферичних артеріальних судин. Система будує цей зв'язок "
    "для кожного учасника, оцінює його **стабільність** у базовому стані та "
    "**реактивність** під час когнітивного й фізичного навантажень."
)

st.markdown(
    "Завантажте дані ДМАТ, щоб отримати індивідуальний профіль тонусу: "
    "стабільність базового зв'язку, реактивність під навантаженням та "
    "бінарний флаг реактивного тонусу."
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
        st.success(f"**{uploaded_file.name}** ({uploaded_file.size:,} байт)")
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
            st.error(f"Помилка перевірки даних: {exc}")
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
        summary_view = pipeline.build_summary_view(res_df)

        # Stage 4 — figures
        st.write("Побудова графіків…")
        demographics_table = pipeline.create_demographics_table(df)
        figures = pipeline.generate_figures(df, res_df)

        status.update(label="Аналіз завершено!", state="complete", expanded=False)

    st.session_state.results = PipelineResults(
        subject_metrics=res_df,
        demographics_table=demographics_table,
        summary_view=summary_view,
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
        with st.expander(f"Відкинуті рядки ({len(excluded)})", expanded=False):
            st.dataframe(excluded, use_container_width=True)
            st.download_button(
                "Завантажити відкинуті рядки",
                excluded.to_csv(index=False),
                file_name="excluded_rows.csv",
                mime="text/csv",
            )

    col1, col2, col3 = st.columns(3)
    col1.metric("Учасники", results.n_subjects)
    col2.metric("Записи", f"{results.n_records:,}")
    col3.metric("Графіки", len(results.figures))

    tab_summary, tab_metrics, tab_figures = st.tabs(
        ["Підсумок", "Профіль тонусу по учасниках", "Графіки"]
    )

    # --- Summary tab ---
    with tab_summary:
        st.subheader("Огляд вимірювань за станами")
        st.dataframe(results.demographics_table, use_container_width=True, hide_index=True)

        st.subheader("Оцінка тонусу по когорті")

        st.markdown("**Форма зв'язку ДАТ ↔ (САТ, ЧСС) у базовому стані**")
        st.caption(
            "Для кожного учасника обирається модель, яка найкраще описує ДАТ як функцію "
            "САТ і ЧСС. Розмаїття виграшних форм вказує, що структура тонусу "
            "неоднорідна між учасниками."
        )
        st.dataframe(results.summary_view.model_counts, use_container_width=True, hide_index=True)

        st.markdown("**Стабільність базового зв'язку (MAE)**")
        st.caption(
            "Менший MAE — тісніший зв'язок ДАТ ↔ (САТ, ЧСС), тобто стабільніший "
            "базовий тонус. Значення > 10 мм рт. ст. свідчать про слабкий зв'язок."
        )
        baseline = results.summary_view.baseline_stats
        baseline_cols = st.columns(4)
        baseline_cols[0].metric("Учасники (n)", baseline["n"])
        baseline_cols[1].metric("Медіана MAE", f"{baseline['median']:.2f} мм рт. ст.")
        baseline_cols[2].metric(
            "IQR",
            f"{baseline['q25']:.2f} – {baseline['q75']:.2f} мм рт. ст.",
        )
        baseline_cols[3].metric(
            "Діапазон",
            f"{baseline['min']:.2f} – {baseline['max']:.2f} мм рт. ст.",
        )

        st.markdown("**Реактивність тонусу за умовами**")
        st.caption(
            "Зростання MAE та зміщення залишків під час когнітивного чи фізичного "
            "навантаження відображає порушення базового зв'язку ДАТ ↔ (САТ, ЧСС) — "
            "тобто реактивну зміну тонусу."
        )
        st.dataframe(results.summary_view.condition_stats, use_container_width=True, hide_index=True)

        st.markdown(
            f"**Флаг реактивного тонусу** (когнітивне навантаження): "
            f"респонденти `n={results.summary_view.n_responders}`, "
            f"нереспонденти `n={results.summary_view.n_non_responders}`"
        )
        st.caption(
            "Респондент — учасник, у якого зв'язок ДАТ ↔ (САТ, ЧСС) значимо "
            "порушується під когнітивним навантаженням (зростання MAE понад 50% "
            "або зміщення залишків понад 2 мм рт. ст.)."
        )
        st.dataframe(results.summary_view.subgroup_stats, use_container_width=True, hide_index=True)

        with st.expander("Сирий текстовий звіт", expanded=False):
            st.code(results.summary_text, language=None)

    # --- Subject metrics tab ---
    with tab_metrics:
        st.subheader("Індивідуальний профіль тонусу")
        st.caption(
            "Для кожного учасника: виграшна форма зв'язку, стабільність базового "
            "тонусу (референтний MAE) та показники реактивності під навантаженнями "
            "(MAE, зміщення, аномалія у % відносно базового MAE)."
        )
        display_pipeline = WebPipeline()
        st.dataframe(
            display_pipeline.localize_subject_metrics(results.subject_metrics),
            use_container_width=True,
        )

        st.download_button(
            "Завантажити CSV",
            results.subject_metrics.to_csv(index=False),
            file_name="profil_tonusu_po_uchasnykakh.csv",
            mime="text/csv",
        )

    # --- Figures tab ---
    with tab_figures:
        figure_titles = {
            "dotplots": "Точкові графіки",
            "obs_vs_pred": "Спостережуваний і прогнозований ДАТ",
            "timeseries_residuals": "Часові ряди та залишки",
        }
        for name, fig in results.figures.items():
            st.subheader(figure_titles.get(name, name.replace("_", " ").title()))
            st.pyplot(fig)
