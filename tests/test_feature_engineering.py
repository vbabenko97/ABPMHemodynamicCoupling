import numpy as np
import pandas as pd

from src.config import Columns, Config
from src.feature_engineering import DBPFeatureExtractor


def test_dbp_feature_extractor_returns_expected_six_features() -> None:
    extractor = DBPFeatureExtractor(Config())
    df = pd.DataFrame(
        {
            Columns.SBP: [120.0],
            Columns.HR: [60.0],
        }
    )

    features = extractor.extract(df)

    assert features.shape == (1, 6)
    np.testing.assert_allclose(
        features[0],
        np.array([120.0, 60.0, 1 / 120.0, 1 / 60.0, 7200.0, 1 / 7200.0]),
    )
