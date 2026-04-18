"""
Synthetic stress and workload simulator.

This module provides a simple latent-state simulator for demo and UI prototyping.
It is intentionally illustrative and should not be treated as a physiological
or clinical model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PersonProfile:
    """Profile parameters that shape baseline synthetic physiology."""

    age: int
    sex: str
    fitness: int
    trait_anxiety: float
    caffeine_cups: int
    smoker: bool
    on_beta_blocker: bool

    def baselines(self) -> dict[str, float]:
        """Return illustrative baseline priors for the simulated channels."""
        age_delta = self.age - 30
        sex_hr = 2.0 if self.sex == "female" else 0.0
        sex_sbp = 1.5 if self.sex == "male" else 0.0
        sex_dbp = 1.0 if self.sex == "male" else 0.0

        hr = (
            72.0
            + 0.08 * age_delta
            - 2.5 * (self.fitness - 3)
            + 1.25 * self.caffeine_cups
            + 6.0 * self.trait_anxiety
            + sex_hr
            - (8.0 if self.on_beta_blocker else 0.0)
            + (1.5 if self.smoker else 0.0)
        )
        sbp = (
            116.0
            + 0.35 * age_delta
            - 1.2 * (self.fitness - 3)
            + 1.5 * self.caffeine_cups
            + 4.0 * self.trait_anxiety
            + sex_sbp
            + (2.0 if self.smoker else 0.0)
        )
        dbp = (
            74.0
            + 0.12 * age_delta
            - 0.6 * (self.fitness - 3)
            + 0.9 * self.caffeine_cups
            + 2.0 * self.trait_anxiety
            + sex_dbp
            + (1.0 if self.smoker else 0.0)
        )

        return {
            "HR": float(np.clip(hr, 48.0, 98.0)),
            "SBP": float(np.clip(sbp, 95.0, 145.0)),
            "DBP": float(np.clip(dbp, 55.0, 95.0)),
        }


@dataclass(slots=True)
class EventTemplate:
    """Template for a latent-state perturbation."""

    label: str
    duration_range_s: tuple[int, int]
    arousal_input: float = 0.0
    workload_input: float = 0.0
    physical_input: float = 0.0
    fatigue_input: float = 0.0


@dataclass(slots=True)
class ActiveEvent:
    """Currently active event instance."""

    label: str
    remaining_s: int
    arousal_input: float
    workload_input: float
    physical_input: float
    fatigue_input: float

    def tick(self) -> bool:
        """Advance the event and report whether it remains active."""
        self.remaining_s -= 1
        return self.remaining_s > 0


class OnlineBaseline:
    """Warmup plus guarded EWMA baseline for one channel."""

    def __init__(self, warmup_seconds: int = 60, alpha: float = 0.02) -> None:
        self.warmup_seconds = warmup_seconds
        self.alpha = alpha
        self._warmup: list[float] = []
        self.ready = False
        self.mu = 0.0
        self.var = 1.0

    def observe(self, value: float) -> None:
        """Collect warmup observations until the baseline becomes usable."""
        if self.ready:
            return

        self._warmup.append(float(value))
        if len(self._warmup) >= self.warmup_seconds:
            arr = np.asarray(self._warmup, dtype=float)
            self.mu = float(arr.mean())
            self.var = float(max(arr.var(ddof=1), 1.0))
            self.ready = True

    def score(self, value: float) -> float:
        """Return z-score relative to the current baseline."""
        if not self.ready:
            return 0.0

        sigma = math.sqrt(max(self.var, 1e-6))
        return (float(value) - self.mu) / sigma

    def maybe_update(self, value: float, allow_update: bool) -> None:
        """Adapt the baseline only during relatively quiet periods."""
        if not self.ready or not allow_update:
            return

        delta = float(value) - self.mu
        self.mu = (1.0 - self.alpha) * self.mu + self.alpha * float(value)
        self.var = (1.0 - self.alpha) * self.var + self.alpha * (delta**2)
        self.var = float(max(self.var, 0.25))

    @property
    def sigma(self) -> float:
        """Return the current standard deviation estimate."""
        return math.sqrt(max(self.var, 1e-6))


class PhysiologySimulator:
    """Latent-state simulator for synthetic physiology and response scores."""

    EVENT_LIBRARY: dict[str, EventTemplate] = {
        "deadline": EventTemplate(
            label="Дедлайн / тиск часу",
            duration_range_s=(30, 90),
            workload_input=0.030,
            arousal_input=0.014,
            fatigue_input=0.003,
        ),
        "interruption": EventTemplate(
            label="Переривання / мультизадачність",
            duration_range_s=(10, 40),
            workload_input=0.025,
            arousal_input=0.022,
        ),
        "alarm": EventTemplate(
            label="Сигнал / стартл-відповідь",
            duration_range_s=(4, 12),
            arousal_input=0.085,
        ),
        "social": EventTemplate(
            label="Соціальне оцінювання",
            duration_range_s=(20, 80),
            arousal_input=0.032,
            workload_input=0.018,
        ),
        "physical": EventTemplate(
            label="Фізичний рух",
            duration_range_s=(20, 75),
            physical_input=0.055,
            arousal_input=0.006,
        ),
        "recovery": EventTemplate(
            label="Відновлення / мікропауза",
            duration_range_s=(20, 120),
            arousal_input=-0.020,
            workload_input=-0.020,
            physical_input=-0.010,
            fatigue_input=-0.010,
        ),
    }

    def __init__(
        self,
        profile: PersonProfile,
        *,
        warmup_seconds: int = 60,
        seed: int = 42,
        random_events_enabled: bool = True,
        event_rate_per_min: float = 0.4,
    ) -> None:
        self.profile = profile
        self.baselines = profile.baselines()
        self.rng = np.random.default_rng(seed)
        self.random_events_enabled = random_events_enabled
        self.event_rate_per_min = float(event_rate_per_min)
        self.active_events: list[ActiveEvent] = []
        self.event_log: list[dict[str, int | str]] = []

        self.t = 0
        self.arousal = 0.10 + 0.05 * profile.trait_anxiety
        self.workload = 0.10
        self.physical = 0.02
        self.fatigue = 0.05

        self.hr_noise = 0.0
        self.sbp_noise = 0.0
        self.dbp_noise = 0.0

        self.hr_baseline = OnlineBaseline(warmup_seconds=warmup_seconds)
        self.sbp_baseline = OnlineBaseline(warmup_seconds=warmup_seconds)
        self.dbp_baseline = OnlineBaseline(warmup_seconds=warmup_seconds)

        self.prev_score = 0.0
        self.elevated_count = 0

    def spawn_event(self, key: str) -> None:
        """Create an active event from a template."""
        template = self.EVENT_LIBRARY[key]
        duration = int(
            self.rng.integers(template.duration_range_s[0], template.duration_range_s[1] + 1)
        )
        event = ActiveEvent(
            label=template.label,
            remaining_s=duration,
            arousal_input=template.arousal_input,
            workload_input=template.workload_input,
            physical_input=template.physical_input,
            fatigue_input=template.fatigue_input,
        )
        self.active_events.append(event)
        self.event_log.append({"t": self.t, "event": template.label, "duration_s": duration})

    def maybe_spawn_random_event(self) -> None:
        """Spawn a random event according to the configured rate."""
        if not self.random_events_enabled:
            return

        if self.rng.uniform() >= self.event_rate_per_min / 60.0:
            return

        keys = ["deadline", "interruption", "alarm", "social", "physical", "recovery"]
        weights = np.array([0.18, 0.22, 0.12, 0.14, 0.18, 0.16], dtype=float)
        weights = weights / weights.sum()
        self.spawn_event(str(self.rng.choice(keys, p=weights)))

    def _decay_states(self) -> None:
        self.arousal *= 0.93
        self.workload *= 0.965
        self.physical *= 0.86
        self.fatigue *= 0.995

    def _apply_events(self) -> None:
        surviving: list[ActiveEvent] = []
        for event in self.active_events:
            self.arousal += event.arousal_input
            self.workload += event.workload_input
            self.physical += event.physical_input
            self.fatigue += event.fatigue_input
            if event.tick():
                surviving.append(event)
        self.active_events = surviving

    def _update_fatigue(self) -> None:
        self.fatigue += 0.0025 * max(self.workload - 0.2, 0.0)
        if not self.active_events and self.workload < 0.2 and self.physical < 0.15:
            self.fatigue -= 0.003

    def _clip_states(self) -> None:
        self.arousal = float(np.clip(self.arousal, 0.0, 1.0))
        self.workload = float(np.clip(self.workload, 0.0, 1.0))
        self.physical = float(np.clip(self.physical, 0.0, 1.0))
        self.fatigue = float(np.clip(self.fatigue, 0.0, 1.0))

    def _sample_measurements(self) -> dict[str, float]:
        self.hr_noise = 0.72 * self.hr_noise + self.rng.normal(0.0, 1.0)
        self.sbp_noise = 0.80 * self.sbp_noise + self.rng.normal(0.0, 1.4)
        self.dbp_noise = 0.82 * self.dbp_noise + self.rng.normal(0.0, 1.0)

        circadian_like = 0.8 * math.sin(self.t / 180.0)

        hr = (
            self.baselines["HR"]
            + 13.0 * self.arousal
            + 7.0 * self.workload
            + 24.0 * self.physical
            + 4.0 * self.fatigue
            + circadian_like
            + self.hr_noise
        )
        sbp = (
            self.baselines["SBP"]
            + 11.0 * self.arousal
            + 4.0 * self.workload
            + 16.0 * self.physical
            + 3.0 * self.fatigue
            + 0.7 * circadian_like
            + self.sbp_noise
        )
        dbp = (
            self.baselines["DBP"]
            + 6.0 * self.arousal
            + 2.0 * self.workload
            + 9.0 * self.physical
            + 2.0 * self.fatigue
            + 0.5 * circadian_like
            + self.dbp_noise
        )

        return {
            "HR": float(np.clip(hr, 42.0, 170.0)),
            "SBP": float(np.clip(sbp, 85.0, 210.0)),
            "DBP": float(np.clip(dbp, 45.0, 130.0)),
        }

    def step(self) -> dict[str, float | str]:
        """Advance the simulator by one second and return the observation row."""
        self.t += 1
        self.maybe_spawn_random_event()
        self._decay_states()
        self._apply_events()
        self._update_fatigue()
        self._clip_states()

        obs = self._sample_measurements()
        self.hr_baseline.observe(obs["HR"])
        self.sbp_baseline.observe(obs["SBP"])
        self.dbp_baseline.observe(obs["DBP"])

        z_hr = max(self.hr_baseline.score(obs["HR"]), 0.0)
        z_sbp = max(self.sbp_baseline.score(obs["SBP"]), 0.0)
        z_dbp = max(self.dbp_baseline.score(obs["DBP"]), 0.0)
        score = 0.50 * z_hr + 0.30 * z_sbp + 0.20 * z_dbp

        if score >= 2.0:
            self.elevated_count += 1
        else:
            self.elevated_count = max(self.elevated_count - 1, 0)

        status = "warmup"
        if self.hr_baseline.ready and self.sbp_baseline.ready and self.dbp_baseline.ready:
            if self.elevated_count >= 3 and score >= 2.5:
                status = "stress_response"
            elif score >= 1.5:
                status = "elevated"
            else:
                status = "baseline"

        quiet = self.prev_score < 1.1 and self.physical < 0.35
        self.hr_baseline.maybe_update(obs["HR"], allow_update=quiet)
        self.sbp_baseline.maybe_update(obs["SBP"], allow_update=quiet)
        self.dbp_baseline.maybe_update(obs["DBP"], allow_update=quiet)
        self.prev_score = score

        active_labels = ", ".join(event.label for event in self.active_events)
        return {
            "t": self.t,
            **obs,
            "HR_baseline": self.hr_baseline.mu if self.hr_baseline.ready else np.nan,
            "SBP_baseline": self.sbp_baseline.mu if self.sbp_baseline.ready else np.nan,
            "DBP_baseline": self.dbp_baseline.mu if self.dbp_baseline.ready else np.nan,
            "HR_upper_band": (
                self.hr_baseline.mu + 2.0 * self.hr_baseline.sigma
                if self.hr_baseline.ready
                else np.nan
            ),
            "SBP_upper_band": (
                self.sbp_baseline.mu + 2.0 * self.sbp_baseline.sigma
                if self.sbp_baseline.ready
                else np.nan
            ),
            "DBP_upper_band": (
                self.dbp_baseline.mu + 2.0 * self.dbp_baseline.sigma
                if self.dbp_baseline.ready
                else np.nan
            ),
            "z_hr": z_hr,
            "z_sbp": z_sbp,
            "z_dbp": z_dbp,
            "response_score": score,
            "status": status,
            "arousal_state": self.arousal,
            "workload_state": self.workload,
            "physical_state": self.physical,
            "fatigue_state": self.fatigue,
            "active_events": active_labels,
        }
