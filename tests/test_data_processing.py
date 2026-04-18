import pandas as pd
import pytest

from abpm_hemodynamic_coupling.config import Columns
from abpm_hemodynamic_coupling.data_processing import DataValidator, Labeler


def test_apply_hierarchy_prioritizes_air_alert() -> None:
    row = pd.Series(
        {
            Columns.ALERT: 1,
            Columns.BEFORE: 1,
            Columns.IS_COG: 1,
            Columns.PRE: 1,
        }
    )

    assert Labeler.apply_hierarchy(row) == Columns.LABEL_AIR_ALERT


def test_validate_monitoring_data_accepts_valid_dataframe() -> None:
    df = pd.DataFrame(
        {
            Columns.PAT_ID: [1, 1],
            Columns.TIME: ["2026-01-01 08:00:00", "2026-01-01 08:30:00"],
            Columns.SBP: [120, 118],
            Columns.DBP: [78, 75],
            Columns.HR: [68, 70],
        }
    )

    DataValidator.validate_monitoring_data(df)


def test_validate_monitoring_data_rejects_out_of_range_values() -> None:
    df = pd.DataFrame(
        {
            Columns.PAT_ID: [1],
            Columns.TIME: ["2026-01-01 08:00:00"],
            Columns.SBP: [400],
            Columns.DBP: [80],
            Columns.HR: [70],
        }
    )

    with pytest.raises(ValueError, match="SBP"):
        DataValidator.validate_monitoring_data(df)


def test_sanitize_monitoring_data_drops_out_of_range_rows() -> None:
    df = pd.DataFrame(
        {
            Columns.PAT_ID: [1, 1, 1],
            Columns.TIME: [
                "2026-01-01 08:00:00",
                "2026-01-01 08:30:00",
                "2026-01-01 09:00:00",
            ],
            Columns.SBP: [120, 400, 118],
            Columns.DBP: [78, 80, 20],
            Columns.HR: [68, 70, 71],
        }
    )

    cleaned, removed_n = DataValidator.sanitize_monitoring_data(df)

    assert removed_n == 2
    assert len(cleaned) == 1
    assert cleaned.iloc[0][Columns.SBP] == 120
