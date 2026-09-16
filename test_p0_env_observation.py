"""Regression checks for the P0 single-step PPO observation contract."""

import numpy as np

from rl.envs.flow_env import DeepFlowEnv


def main() -> None:
    env = DeepFlowEnv(config_dir="configs", domain_randomization=False, episode_len=1)
    observation, _ = env.reset(options={
        "bandwidth_mbps": 1.0,
        "latency_ms": 50.0,
        "prompt_len": 512,
    })
    assert observation.shape == (3,)
    assert observation.tolist() == [1.0, 50.0, 512.0]

    next_observation, _, _, truncated, info = env.step([0, 1, 0])
    assert truncated
    assert np.array_equal(next_observation, observation)
    assert info["mode"] == "Token Speculative DeepFlow"

    _, _, _, _, remote_info = env.step([0, 0, 0])
    assert remote_info["mode"] == "Remote Target without Speculation"
    print("P0 environment observation regression checks passed.")


if __name__ == "__main__":
    main()
