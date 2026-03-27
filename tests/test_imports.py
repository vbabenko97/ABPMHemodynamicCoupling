def test_src_package_exports_core_symbols() -> None:
    from src import Columns, Config, ModelPerformance, StatisticalResult, SubjectResult

    assert Config is not None
    assert Columns is not None
    assert ModelPerformance is not None
    assert StatisticalResult is not None
    assert SubjectResult is not None


def test_run_pipeline_module_imports_without_sys_path_hack() -> None:
    import run_pipeline

    assert hasattr(run_pipeline, "main")
