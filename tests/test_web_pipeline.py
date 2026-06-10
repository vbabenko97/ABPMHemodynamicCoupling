import pandas as pd

from abpm_hemodynamic_coupling.config import Columns
from abpm_hemodynamic_coupling.web_pipeline import WebPipeline


def _frame_with_non_numeric_sbp() -> pd.DataFrame:
    """An uploaded frame where one SBP cell is non-numeric text."""
    return pd.DataFrame(
        {
            Columns.PAT_ID: [1, 1, 1],
            Columns.TIME: [
                "2026-01-01 08:00:00",
                "2026-01-01 08:30:00",
                "2026-01-01 09:00:00",
            ],
            Columns.SBP: [120, "bad", 118],
            Columns.DBP: [78, 80, 75],
            Columns.HR: [68, 70, 71],
        }
    )


def test_sanitize_drops_non_numeric_rows_and_returns_numeric_columns() -> None:
    """sanitize() should drop rows with non-numeric physiological cells and
    return numeric-dtype columns, not let the bad cell survive."""
    pipeline = WebPipeline()
    cleaned, report = pipeline.sanitize(_frame_with_non_numeric_sbp())

    assert len(cleaned) == 2
    assert report.counts.get(Columns.SBP) == 1
    assert pd.api.types.is_numeric_dtype(cleaned[Columns.SBP])


def test_sanitized_frame_passes_validation() -> None:
    """A sanitized upload must validate without raising; previously the
    non-numeric cell survived sanitize and tripped the numeric-dtype check."""
    pipeline = WebPipeline()
    cleaned, _ = pipeline.sanitize(_frame_with_non_numeric_sbp())

    processed = pipeline.validate_and_preprocess(cleaned)
    assert len(processed) == 2
