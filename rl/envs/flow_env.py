# rl/envs/flow_env.py
import json
import os
from typing import Optional, Dict, Any, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.hardware import Device, NetworkLink
from core.model_spec import LLaMAModel
from engine.simulator import DeepFlowSimulator


class DeepFlowEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config_dir: str = "configs",
        total_batch_size: int = 32,
        episode_len: int = 1,
        domain_randomization: bool = True,
        seed: Optional[int] = None,
        reward_mode: str = "shaped",
    ):
        super().__init__()

        self.config_dir = config_dir
        self.total_batch_size = total_batch_size
        self.episode_len = episode_len
        self.domain_randomization = domain_randomization
        self.reward_mode = reward_mode

        self.rng = np.random.default_rng(seed)

        self._load_physical_world()

        self.mb_options = [1, 2, 4, 8, 16, 32]
        self.k_options = [0, 1, 3, 5, 7, 10]
        self.num_layers = self.target_model.config["num_layers"]

        self.action_space = spaces.MultiDiscrete([
            len(self.mb_options),
            len(self.k_options),
            self.num_layers + 1,
        ])

        self.observation_space = spaces.Box(
            low=np.array([0.1, 1.0, 1.0, 0.0], dtype=np.float32),
            high=np.array([1000.0, 1000.0, float(self.max_analysis_prompt_len), 10000.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.current_bandwidth = float(self.network.bandwidth_mbps)
        self.current_latency = float(self.network.latency_s * 1000.0)
        self.current_prompt_len = 512
        self.last_throughput = 0.0
        self.step_count = 0

    def _load_physical_world(self):
        with open(os.path.join(self.config_dir, "devices_paper.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)

        dev_cfg = cfg["devices"]
        net_cfg = cfg["network"]
        sim_cfg = cfg.get("simulator", {})

        self.edge = Device(**dev_cfg[0])
        self.cloud = Device(**dev_cfg[1])
        self.network = NetworkLink(**net_cfg)

        self.target_model = LLaMAModel(os.path.join(self.config_dir, "llama2_7b_paper.json"))
        self.draft_model = LLaMAModel(os.path.join(self.config_dir, "llama_1b_paper.json"))

        self.memory_budget_ratio = float(sim_cfg.get("memory_budget_ratio", 0.78))
        self.oom_penalty = float(sim_cfg.get("oom_penalty", -14.0)) if "oom_penalty" in sim_cfg else -14.0
        self.max_analysis_prompt_len = int(sim_cfg.get("max_analysis_prompt_len", 8192))

        self.simulator = DeepFlowSimulator(
            draft_model=self.draft_model,
            target_model=self.target_model,
            edge=self.edge,
            cloud=self.cloud,
            network=self.network,
            memory_budget_ratio=float(sim_cfg.get("memory_budget_ratio", 0.78)),
            weight_reservation_factor=float(sim_cfg.get("weight_reservation_factor", 1.08)),
            activation_safety_factor=float(sim_cfg.get("activation_safety_factor", 2.40)),
            kv_cache_safety_factor=float(sim_cfg.get("kv_cache_safety_factor", 1.35)),
            framework_overhead_edge_mb=float(sim_cfg.get("framework_overhead_edge_mb", 2500.0)),
            framework_overhead_cloud_mb=float(sim_cfg.get("framework_overhead_cloud_mb", 6000.0)),
            comm_buffer_safety_factor=float(sim_cfg.get("comm_buffer_safety_factor", 2.0)),
            verify_workspace_factor=float(sim_cfg.get("verify_workspace_factor", 2.2)),
        )

    # ------------------------------------------------------------------
    # Scenario helpers
    # ------------------------------------------------------------------
    def _sync_network(self):
        self.network.bandwidth_mbps = float(self.current_bandwidth)
        self.network.latency_s = float(self.current_latency) / 1000.0
        raw_gb_s = (self.current_bandwidth / 8.0) / 1024.0
        self.network.bandwidth_gb_s = raw_gb_s * self.network.bandwidth_efficiency

    def set_network_conditions(self, bandwidth_mbps: float, latency_ms: float):
        self.current_bandwidth = float(bandwidth_mbps)
        self.current_latency = float(latency_ms)
        self._sync_network()

    def set_prompt_length(self, prompt_len: int):
        self.current_prompt_len = int(prompt_len)

    def set_scenario(self, bandwidth_mbps: float, latency_ms: float, prompt_len: int):
        self.current_bandwidth = float(bandwidth_mbps)
        self.current_latency = float(latency_ms)
        self.current_prompt_len = int(prompt_len)
        self._sync_network()

    def _sample_random_scenario(self):
        bw = float(np.exp(self.rng.uniform(np.log(0.5), np.log(100.0))))
        lat = float(self.rng.uniform(5.0, 120.0))
        prompt_candidates = [128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
        prompt_len = int(self.rng.choice(prompt_candidates))
        self.set_scenario(bw, lat, prompt_len)

    def _get_obs(self):
        return np.array([
            self.current_bandwidth,
            self.current_latency,
            self.current_prompt_len,
            self.last_throughput,
        ], dtype=np.float32)

    def reset(self, seed=None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.last_throughput = 0.0
        options = options or {}

        if "bandwidth_mbps" in options and "latency_ms" in options and "prompt_len" in options:
            self.set_scenario(
                bandwidth_mbps=options["bandwidth_mbps"],
                latency_ms=options["latency_ms"],
                prompt_len=options["prompt_len"],
            )
        elif self.domain_randomization:
            self._sample_random_scenario()
        else:
            self.set_scenario(
                bandwidth_mbps=self.current_bandwidth,
                latency_ms=self.current_latency,
                prompt_len=512,
            )

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------
    @property
    def num_discrete_actions(self) -> int:
        return len(self.mb_options) * len(self.k_options) * (self.num_layers + 1)

    def flatten_action(self, action: List[int]) -> int:
        mb_idx, k_idx, part = [int(x) for x in action]
        n_k = len(self.k_options)
        n_p = self.num_layers + 1
        return mb_idx * n_k * n_p + k_idx * n_p + part

    def unflatten_action(self, idx: int) -> List[int]:
        n_k = len(self.k_options)
        n_p = self.num_layers + 1
        mb_idx = idx // (n_k * n_p)
        rem = idx % (n_k * n_p)
        k_idx = rem // n_p
        part = rem % n_p
        return [int(mb_idx), int(k_idx), int(part)]

    def get_valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.num_discrete_actions, dtype=bool)
        for idx in range(self.num_discrete_actions):
            action = self.unflatten_action(idx)
            info = self.evaluate_action(action)
            mask[idx] = bool(info["valid"] and info["feasible"])
        return mask

    # ------------------------------------------------------------------
    # Unified evaluation
    # ------------------------------------------------------------------
    def evaluate_action(self, action):
        mb_idx, k_idx, partition_point = [int(x) for x in action]

        micro_batch_size = self.mb_options[mb_idx]
        k_steps = self.k_options[k_idx]

        if self.total_batch_size % micro_batch_size != 0:
            return {
                "valid": False,
                "feasible": False,
                "error": "Invalid micro-batch size",
                "throughput": 0.0,
                "makespan": 1e9,
                "data_size_mb": 0.0,
                "partition_point": partition_point,
                "k_steps": k_steps,
                "micro_batch_size": micro_batch_size,
                "edge_peak_memory_mb": 0.0,
                "cloud_peak_memory_mb": 0.0,
                "edge_budget_mb": self.edge.available_memory_gb * 1024.0 * self.memory_budget_ratio,
                "cloud_budget_mb": self.cloud.available_memory_gb * 1024.0 * self.memory_budget_ratio,
                "oom_device": "invalid_micro_batch",
                "memory_breakdown": {},
                "util_edge": 0.0,
                "util_cloud": 0.0,
                "bottleneck": -1,
            }

        result = self.simulator.simulate(
            total_batch_size=self.total_batch_size,
            micro_batch_size=micro_batch_size,
            k_steps=k_steps,
            partition_point=partition_point,
            prompt_len=self.current_prompt_len,
        )

        return {
            "valid": True,
            "feasible": result.feasible,
            "micro_batch_size": micro_batch_size,
            "k_steps": k_steps,
            "partition_point": partition_point,
            "throughput": result.throughput,
            "makespan": result.makespan,
            "data_size_mb": result.data_size_mb,
            "bottleneck": result.bottleneck_stage,
            "effective_tokens_per_seq": result.effective_tokens_per_seq,
            "total_effective_tokens": result.total_effective_tokens,
            "util_edge": result.util_edge,
            "util_cloud": result.util_cloud,
            "bubble_rate": result.bubble_rate,
            "timeline": result.timeline,
            "stage_costs": result.stage_costs,
            "verify_seq_len": result.verify_seq_len,
            "edge_peak_memory_mb": result.edge_peak_memory_mb,
            "cloud_peak_memory_mb": result.cloud_peak_memory_mb,
            "edge_budget_mb": result.edge_budget_mb,
            "cloud_budget_mb": result.cloud_budget_mb,
            "oom_device": result.oom_device,
            "memory_breakdown": result.memory_breakdown,
        }

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_reward(self, info: Dict[str, Any]) -> float:
        if not info["valid"]:
            return -12.0

        if not info["feasible"]:
            return float(self.oom_penalty)

        throughput = float(info["throughput"])
        part = int(info["partition_point"])

        base = np.log(throughput + 1e-5)

        if self.reward_mode == "raw":
            return float(base)

        bonus = 0.0
        penalty = 0.0

        # 只保留非常轻的结构先验：鼓励 DeepFlow 主模式，但不强行压制其他可行动作
        if part == 0:
            bonus += 0.08
        else:
            penalty += 0.01 * min(part, 8)

        # 通信开销轻惩罚
        penalty += 0.02 * np.log1p(float(info["data_size_mb"]))

        # 靠近显存边界时轻惩罚，鼓励在可行域内留一定安全裕量
        edge_ratio = float(info["edge_peak_memory_mb"]) / max(float(info["edge_budget_mb"]), 1e-6)
        cloud_ratio = float(info["cloud_peak_memory_mb"]) / max(float(info["cloud_budget_mb"]), 1e-6)

        penalty += 0.25 * max(0.0, edge_ratio - 0.80)
        penalty += 0.25 * max(0.0, cloud_ratio - 0.80)

        reward = base + bonus - penalty
        return float(reward)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action):
        info = self.evaluate_action(action)

        throughput = float(info["throughput"]) if (info["valid"] and info["feasible"]) else 0.0
        self.last_throughput = throughput

        reward = self._compute_reward(info)

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.episode_len

        return self._get_obs(), reward, terminated, truncated, info