"""Public, reproducible uncertainty envelopes for simulator stress experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class StressProfile:
    """A parameterized uncertainty envelope, not a deployment measurement model."""

    name: str
    protocol_overhead_ms: float
    jitter_std_ms: float
    packet_loss_rate: float
    runtime_slowdown: float
    fragmentation_reserve: float
    kv_block_size: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Stress profile name must be non-empty")
        if self.protocol_overhead_ms < 0.0 or self.jitter_std_ms < 0.0:
            raise ValueError("Network overhead and jitter must be non-negative")
        if not 0.0 <= self.packet_loss_rate <= 1.0:
            raise ValueError("packet_loss_rate must be in [0, 1]")
        if self.runtime_slowdown <= 0.0:
            raise ValueError("runtime_slowdown must be positive")
        if not 0.0 <= self.fragmentation_reserve < 1.0:
            raise ValueError("fragmentation_reserve must be in [0, 1)")
        if self.kv_block_size <= 0:
            raise ValueError("kv_block_size must be positive")

    @property
    def protocol_overhead_s(self) -> float:
        return self.protocol_overhead_ms / 1000.0

    @property
    def is_stochastic_network(self) -> bool:
        return self.jitter_std_ms > 0.0 or self.packet_loss_rate > 0.0

    def as_metadata(self) -> Dict[str, float | int | str]:
        return {
            "name": self.name,
            "protocol_overhead_ms": self.protocol_overhead_ms,
            "jitter_std_ms": self.jitter_std_ms,
            "packet_loss_rate": self.packet_loss_rate,
            "runtime_slowdown": self.runtime_slowdown,
            "fragmentation_reserve": self.fragmentation_reserve,
            "kv_block_size": self.kv_block_size,
        }


@dataclass(frozen=True)
class NetworkTransmissionSample:
    """One public random draw for a single micro-batch transmission."""

    jitter_s: float
    retry_jitter_s: float
    packet_lost: bool

    def __post_init__(self) -> None:
        if self.jitter_s < 0.0 or self.retry_jitter_s < 0.0:
            raise ValueError("Network jitter draws must be non-negative")


BASE_STRESS_PROFILE = StressProfile(
    name="base",
    protocol_overhead_ms=0.5,
    jitter_std_ms=0.0,
    packet_loss_rate=0.0,
    runtime_slowdown=1.0,
    fragmentation_reserve=0.0,
    kv_block_size=1,
)
MILD_STRESS_PROFILE = StressProfile(
    name="mild",
    protocol_overhead_ms=2.0,
    jitter_std_ms=5.0,
    packet_loss_rate=0.01,
    runtime_slowdown=1.10,
    fragmentation_reserve=0.10,
    kv_block_size=16,
)
SEVERE_STRESS_PROFILE = StressProfile(
    name="severe",
    protocol_overhead_ms=5.0,
    jitter_std_ms=15.0,
    packet_loss_rate=0.03,
    runtime_slowdown=1.25,
    fragmentation_reserve=0.20,
    kv_block_size=64,
)
STRESS_PROFILES = {
    profile.name: profile
    for profile in (BASE_STRESS_PROFILE, MILD_STRESS_PROFILE, SEVERE_STRESS_PROFILE)
}


def get_stress_profile(name: str) -> StressProfile:
    try:
        return STRESS_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown stress profile: {name}") from exc


def generate_network_trials(
    profile: StressProfile,
    *,
    random_seed: int,
    trial_count: int,
    transmissions_per_trial: int,
) -> Tuple[Tuple[NetworkTransmissionSample, ...], ...]:
    """Generate common random numbers for all policies in one scenario/profile pair."""
    if trial_count <= 0 or transmissions_per_trial <= 0:
        raise ValueError("trial_count and transmissions_per_trial must be positive")

    rng = np.random.default_rng(random_seed)
    jitter_std_s = profile.jitter_std_ms / 1000.0
    trials = []
    for _ in range(trial_count):
        transmissions = []
        for _ in range(transmissions_per_trial):
            jitter_s = max(0.0, float(rng.normal(0.0, jitter_std_s)))
            retry_jitter_s = max(0.0, float(rng.normal(0.0, jitter_std_s)))
            packet_lost = bool(rng.random() < profile.packet_loss_rate)
            transmissions.append(NetworkTransmissionSample(
                jitter_s=jitter_s,
                retry_jitter_s=retry_jitter_s,
                packet_lost=packet_lost,
            ))
        trials.append(tuple(transmissions))
    return tuple(trials)
