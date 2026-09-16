"""Frozen baseline policies and oracle searches backed by the unified environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


Scenario = Tuple[str, float, float, int]

GLOBAL_STATIC_CALIBRATION_SCENARIOS: Tuple[Scenario, ...] = tuple(
    (
        f"calibration_bw{bandwidth:g}_lat{link_delay:g}_p{prompt_len}",
        bandwidth,
        link_delay,
        prompt_len,
    )
    for bandwidth in (0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0)
    for link_delay in (10.0, 30.0, 50.0, 100.0, 120.0)
    for prompt_len in (128, 512, 1024, 1536)
)

INDEPENDENT_TEST_SCENARIOS: Tuple[Scenario, ...] = tuple(
    (
        f"test_bw{bandwidth:g}_lat{link_delay:g}_p{prompt_len}",
        bandwidth,
        link_delay,
        prompt_len,
    )
    for bandwidth in (0.75, 3.0, 20.0, 75.0)
    for link_delay in (20.0, 80.0)
    for prompt_len in (256, 768, 1280, 1536)
)


@dataclass(frozen=True)
class GlobalStaticSelection:
    action: Tuple[int, int, int]
    mean_oracle_ratio: float
    calibration_scenario_count: int

    def as_metadata(self) -> Dict[str, Any]:
        return {
            "action": list(self.action),
            "mean_oracle_ratio": self.mean_oracle_ratio,
            "calibration_scenario_count": self.calibration_scenario_count,
            "calibration_scenarios": [
                {
                    "scenario_id": scenario_id,
                    "bandwidth_mbps": bandwidth,
                    "link_delay_ms": link_delay,
                    "prompt_len": prompt_len,
                }
                for scenario_id, bandwidth, link_delay, prompt_len in GLOBAL_STATIC_CALIBRATION_SCENARIOS
            ],
        }


def set_scenario(environment: Any, bandwidth_mbps: float, link_delay_ms: float, prompt_len: int) -> None:
    environment.reset(options={
        "bandwidth_mbps": bandwidth_mbps,
        "latency_ms": link_delay_ms,
        "prompt_len": prompt_len,
    })
    environment.set_scenario(bandwidth_mbps, link_delay_ms, prompt_len)


def search_per_scenario_oracle(environment: Any) -> Tuple[Optional[List[int]], Optional[Dict[str, Any]]]:
    """Find the feasible full-action optimum for the current scenario only."""
    best_action: Optional[List[int]] = None
    best_info: Optional[Dict[str, Any]] = None
    best_throughput = -1.0

    for flat_index in range(environment.num_discrete_actions):
        action = environment.unflatten_action(flat_index)
        info = environment.evaluate_action(action)
        if info["valid"] and info["feasible"] and info["throughput"] > best_throughput:
            best_action = action
            best_info = info
            best_throughput = float(info["throughput"])

    return best_action, best_info


def token_deepflow_candidate_actions(environment: Any) -> List[List[int]]:
    """Return all fixed Token-ID speculative actions, excluding remote no-speculation."""
    return [
        [mb_idx, k_idx, 0]
        for mb_idx in range(len(environment.mb_options))
        for k_idx, k_steps in enumerate(environment.k_options)
        if k_steps > 0
    ]


def select_global_static_deepflow(
    environment: Any,
    calibration_scenarios: Sequence[Scenario] = GLOBAL_STATIC_CALIBRATION_SCENARIOS,
) -> GlobalStaticSelection:
    """Choose one Token-ID action on calibration data without using a test scenario."""
    candidates = token_deepflow_candidate_actions(environment)
    if not candidates:
        raise ValueError("No speculative Token-ID action is available for Global Static selection")

    scores = {environment.flatten_action(action): 0.0 for action in candidates}
    for _, bandwidth, link_delay, prompt_len in calibration_scenarios:
        set_scenario(environment, bandwidth, link_delay, prompt_len)
        _, oracle_info = search_per_scenario_oracle(environment)
        oracle_throughput = 0.0 if oracle_info is None else float(oracle_info["throughput"])

        for action in candidates:
            flat_index = environment.flatten_action(action)
            info = environment.evaluate_action(action)
            candidate_throughput = float(info["throughput"]) if info["valid"] and info["feasible"] else 0.0
            ratio = candidate_throughput / oracle_throughput if oracle_throughput > 0.0 else 0.0
            scores[flat_index] += ratio

    divisor = len(calibration_scenarios)
    if divisor == 0:
        raise ValueError("Global Static calibration requires at least one scenario")

    best_flat_index = min(scores)
    best_score = scores[best_flat_index] / divisor
    for flat_index in sorted(scores):
        score = scores[flat_index] / divisor
        if score > best_score + 1e-12:
            best_flat_index = flat_index
            best_score = score

    return GlobalStaticSelection(
        action=tuple(environment.unflatten_action(best_flat_index)),
        mean_oracle_ratio=best_score,
        calibration_scenario_count=divisor,
    )


def heuristic_deepflow_action(environment: Any, link_delay_ms: float, prompt_len: int) -> List[int]:
    """Return the frozen P=0, K=1 heuristic declared before independent testing."""
    if prompt_len <= 256:
        micro_batch_size = 4 if link_delay_ms < 50.0 else 8
    elif prompt_len <= 1024:
        if link_delay_ms < 40.0:
            micro_batch_size = 1
        elif link_delay_ms < 80.0:
            micro_batch_size = 2
        else:
            micro_batch_size = 4
    else:
        micro_batch_size = 1

    return [
        environment.mb_options.index(micro_batch_size),
        environment.k_options.index(1),
        0,
    ]


def token_speculation_without_pipeline_action(environment: Any) -> List[int]:
    """Return the fixed P=0, K=1, MB=32 protocol ablation action."""
    return [
        environment.mb_options.index(32),
        environment.k_options.index(1),
        0,
    ]
