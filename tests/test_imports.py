def test_src_package_exports_core_symbols() -> None:
    from abpm_hemodynamic_coupling import (
        Columns,
        Config,
        ModelPerformance,
        StatisticalResult,
        SubjectResult,
    )

    assert Config is not None
    assert Columns is not None
    assert ModelPerformance is not None
    assert StatisticalResult is not None
    assert SubjectResult is not None


def test_run_pipeline_module_imports_without_sys_path_hack() -> None:
    import run_pipeline

    assert hasattr(run_pipeline, "main")


def test_simulator_module_imports_and_steps() -> None:
    from src.simulator import PersonProfile, PhysiologySimulator

    simulator = PhysiologySimulator(
        PersonProfile(
            age=32,
            sex="male",
            fitness=3,
            trait_anxiety=0.35,
            caffeine_cups=1,
            smoker=False,
            on_beta_blocker=False,
        ),
        warmup_seconds=5,
        seed=42,
        random_events_enabled=False,
    )

    row = simulator.step()

    assert row["t"] == 1
    assert "HR" in row
    assert "SBP" in row
    assert "DBP" in row
    assert "response_score" in row
