import pandas as pd

from abpm_hemodynamic_coupling.config import Config
from abpm_hemodynamic_coupling.models import SubjectResult
from run_pipeline import CohortAnalyzer


def test_subject_result_exports_brandon_metadata() -> None:
    result = SubjectResult(
        participant_id=1,
        train_n=20,
        dbp_winner="Brandon",
        dbp_ref_mae=5.5,
        dbp_cv_mae_brandon=5.5,
        dbp_cv_mae_lasso=5.8,
        dbp_cv_mae_rfe=5.9,
        dbp_cv_mae_ols_sbp=6.1,
        dbp_cv_mae_ols_sbp_hr=5.7,
        dbp_brandon_features="SBP, HR",
        dbp_brandon_feature_count=2,
    )

    exported = result.to_dict()

    assert exported["DBP_CV_MAE_Brandon"] == 5.5
    assert exported["DBP_Brandon_Features"] == "SBP, HR"
    assert exported["DBP_Brandon_Feature_Count"] == 2


def test_build_brandon_feature_counts_aggregates_all_and_winner_subsets() -> None:
    analyzer = CohortAnalyzer(Config())
    res_df = pd.DataFrame(
        [
            {
                "DBP_Winner": "Brandon",
                "DBP_Brandon_Features": "SBP, HR",
                "DBP_Brandon_Feature_Count": 2,
            },
            {
                "DBP_Winner": "OLS(SBP)",
                "DBP_Brandon_Features": "SBP, 1/SBP",
                "DBP_Brandon_Feature_Count": 2,
            },
            {
                "DBP_Winner": "Brandon",
                "DBP_Brandon_Features": "HR",
                "DBP_Brandon_Feature_Count": 1,
            },
        ]
    )

    counts = analyzer._build_brandon_feature_counts(res_df)
    counts = counts.set_index("feature")

    assert counts.loc["SBP", "selected_n_all_subjects"] == 2
    assert counts.loc["SBP", "selected_n_brandon_winners"] == 1
    assert counts.loc["HR", "selected_n_all_subjects"] == 2
    assert counts.loc["HR", "selected_n_brandon_winners"] == 2
    assert counts.loc["SBP*HR", "selected_pct_brandon_winners"] == 0.0
