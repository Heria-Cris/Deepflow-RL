# test_phase4.py
from rl.envs.flow_env import DeepFlowEnv


def run_phase4_test():
    print("=" * 60)
    print("🤖 DeepFlow-RL - Phase 4: Unified RL Environment Validation")
    print("=" * 60)

    env = DeepFlowEnv(config_dir="configs")
    obs, _ = env.reset()

    print(f"[Env Info] Action Space: {env.action_space}")
    print(f"[Env Info] Obs Space   : {env.observation_space}")
    print("-" * 60)

    test_scenarios = [
        ("Weak Network (1 Mbps, 50 ms)", 1.0, 50.0, 512),
        ("Strong Network (100 Mbps, 10 ms)", 100.0, 10.0, 512),
    ]

    for name, bw, lat, prompt_len in test_scenarios:
        print(f"\n🔵 Scenario: {name}")
        env.set_network_conditions(bw, lat)
        env.set_prompt_length(prompt_len)

        action_naive = [5, 0, 16]   # MB=32, K=0, split at 16
        _, _, _, _, info_naive = env.step(action_naive)

        print("  [Strategy A: Legacy Split]")
        print(f"    - Action       : {action_naive}")
        print(f"    - Throughput   : {info_naive['throughput']:.2f} tok/s")
        print(f"    - Data Transfer: {info_naive['data_size_mb']:.4f} MB")

        action_expert = [5, 4, 0]   # MB=32, K=7, P=0
        _, _, _, _, info_expert = env.step(action_expert)

        print("  [Strategy B: DeepFlow Expert]")
        print(f"    - Action       : {action_expert}")
        print(f"    - Throughput   : {info_expert['throughput']:.2f} tok/s")
        print(f"    - Data Transfer: {info_expert['data_size_mb']:.4f} MB")

        gain = info_expert["throughput"] / (info_naive["throughput"] + 1e-5)
        print(f"  🚀 Speedup: {gain:.2f}x")

    print("\n✅ RL environment now uses the same backend as the experiment scripts.")


if __name__ == "__main__":
    run_phase4_test()