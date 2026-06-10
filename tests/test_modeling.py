import numpy as np

from abpm_hemodynamic_coupling.config import Config
from abpm_hemodynamic_coupling.modeling import CrossValidator


def test_cv_tolerates_lasso_convergence_warning() -> None:
    """CV must handle a Lasso ConvergenceWarning the same way final training does.

    With a tiny ``max_iter`` LassoCV cannot converge and emits a
    ConvergenceWarning. The cross-validator must tolerate it (as
    ``ModelTrainer.train`` does) and still record a finite MAE, rather than
    promoting the warning to an error and discarding every fold as NaN -> inf.
    """
    config = Config()
    config.LASSO_MAX_ITER = 1  # force non-convergence -> ConvergenceWarning
    cv = CrossValidator(config)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 6))
    y = rng.normal(size=30)

    perf = cv.evaluate_models(X, y)

    assert perf is not None
    assert np.isfinite(perf.to_dict()["Lasso"])
