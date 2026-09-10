# eval_phase5.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from rl.envs.flow_env import DeepFlowEnv


def build_base_env():
    env = DummyVecEnv([
        lambda: DeepFlowEnv(
            config_dir="configs",
            total_batch_size=32,
            episode_len=1,
            domain_randomization=False,
            seed=123,
            reward_mode="shaped",
        )
    ])
    env = VecMonitor(env)
    return env


def evaluate():
    print("=" * 78)
    print("🕵️ DeepFlow-RL - PPO Evaluation on Memory-Aware Simulator v2")
    print("=" * 78)

    best_model_path = "models/ppo_deepflow/best_model/best_model.zip"
    best_stats_path = "models/ppo_deepflow/best_model/vec_normalize.pkl"
    final_model_path = "models/ppo_deepflow/final_model.zip"
    final_stats_path = "models/ppo_deepflow/vec_normalize.pkl"

    if os.path.exists(best_model_path) and os.path.exists(best_stats_path):
        model_path = best_model_path
        stats_path = best_stats_path
    else:
        model_path = final_model_path
        stats_path = final_stats_path

    base_env = build_base_env()
    env = VecNormalize.load(stats_path, base_env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)

    print(f"✅ Model loaded: {model_path}")
    print(f"✅ VecNormalize loaded: {stats_path}")

    raw_env = env.envs[0]
    mb_options = raw_env.mb_options
    k_options = raw_env.k_options

    scenarios = [
        ("Weak / Short Prompt", 1.0, 50.0, 512),
        ("Moderate / Short Prompt", 5.0, 30.0, 512),
        ("Strong / Short Prompt", 100.0, 10.0, 512),
        ("Weak / Long Prompt", 1.0, 50.0, 2048),
        ("Strong / Long Prompt", 100.0, 10.0, 2048),
        ("Weak / Very Long Prompt", 1.0, 50.0, 4096),
        ("Strong / Very Long Prompt", 100.0, 10.0, 4096),
        ("High RTT / Short Prompt", 10.0, 120.0, 512),
        ("High RTT / Very Long Prompt", 10.0, 120.0, 4096),
    ]

    for name, bw, lat, prompt_len in scenarios:
        print(f"\n🔵 Scenario: {name}")

        raw_env.reset(options={
            "bandwidth_mbps": bw,
            "latency_ms": lat,
            "prompt_len": prompt_len,
        })
        raw_env.set_scenario(bw, lat, prompt_len)

        obs = raw_env._get_obs()
        norm_obs = env.normalize_obs(obs.reshape(1, -1))
        action, _ = model.predict(norm_obs, deterministic=True)
        action = action[0] if len(action.shape) > 1 else action

        info = raw_env.evaluate_action(action)

        mb_idx, k_idx, part = [int(x) for x in action]
        real_mb = mb_options[mb_idx]
        real_k = k_options[k_idx]

        edge_ratio = float(info["edge_peak_memory_mb"]) / max(float(info["edge_budget_mb"]), 1e-6)
        cloud_ratio = float(info["cloud_peak_memory_mb"]) / max(float(info["cloud_budget_mb"]), 1e-6)

        print("  🤖 Agent Decision:")
        print(f"    - Partition Point : {part}")
        print(f"    - Speculative K   : {real_k}")
        print(f"    - Micro-batch     : {real_mb}")

        print("  📈 Performance:")
        print(f"    - Feasible        : {info['feasible']}")
        print(f"    - Throughput      : {info['throughput']:.2f} tok/s")
        print(f"    - Makespan        : {info['makespan']:.4f} s")
        print(f"    - Data Transfer   : {info['data_size_mb']:.4f} MB")
        print(f"    - Bottleneck      : {info['bottleneck']} (0=edge,1=net,2=cloud)")
        print(f"    - Edge Util       : {info['util_edge'] * 100:.1f}%")
        print(f"    - Cloud Util      : {info['util_cloud'] * 100:.1f}%")

        print("  🧠 Memory:")
        print(f"    - Edge Peak/Budget  : {info['edge_peak_memory_mb']:.2f} / {info['edge_budget_mb']:.2f} MB ({edge_ratio:.3f})")
        print(f"    - Cloud Peak/Budget : {info['cloud_peak_memory_mb']:.2f} / {info['cloud_budget_mb']:.2f} MB ({cloud_ratio:.3f})")
        print(f"    - OOM Device        : {info['oom_device']}")

        if part == 0:
            print("  ✅ Strategy: DeepFlow mode (Token-ID transfer)")
        else:
            print("  ⚠️ Strategy: Legacy split / activation transfer")


if __name__ == "__main__":
    evaluate()