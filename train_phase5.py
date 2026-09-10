# train_phase5.py
import os
import numpy as np
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize

from rl.envs.flow_env import DeepFlowEnv


class StrategyLoggerCallback(BaseCallback):
    def __init__(self, log_freq: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.partition_choices = []
        self.k_choices = []
        self.mb_choices = []

    def _on_step(self) -> bool:
        actions = self.locals.get("actions", None)
        if actions is None:
            return True

        for action in np.array(actions):
            self.mb_choices.append(int(action[0]))
            self.k_choices.append(int(action[1]))
            self.partition_choices.append(int(action[2]))

        if self.n_calls % self.log_freq == 0 and len(self.partition_choices) >= self.log_freq:
            avg_part = float(np.mean(self.partition_choices[-self.log_freq:]))
            avg_k = float(np.mean(self.k_choices[-self.log_freq:]))
            avg_mb = float(np.mean(self.mb_choices[-self.log_freq:]))

            self.logger.record("strategy/avg_partition_point", avg_part)
            self.logger.record("strategy/avg_speculative_k_idx", avg_k)
            self.logger.record("strategy/avg_mb_idx", avg_mb)

            print(
                f"[{self.num_timesteps} timesteps] "
                f"avg_part={avg_part:.2f}, avg_k_idx={avg_k:.2f}, avg_mb_idx={avg_mb:.2f}"
            )
        return True


class FeasibilityLoggerCallback(BaseCallback):
    def __init__(self, log_freq: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

        self._window_feasible = []
        self._window_oom = []
        self._window_edge_ratio = []
        self._window_cloud_ratio = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is None:
            return True

        for info in infos:
            feasible = bool(info.get("feasible", False))
            self._window_feasible.append(1.0 if feasible else 0.0)
            self._window_oom.append(0.0 if feasible else 1.0)

            edge_peak = float(info.get("edge_peak_memory_mb", 0.0))
            edge_budget = max(float(info.get("edge_budget_mb", 1e-6)), 1e-6)
            cloud_peak = float(info.get("cloud_peak_memory_mb", 0.0))
            cloud_budget = max(float(info.get("cloud_budget_mb", 1e-6)), 1e-6)

            self._window_edge_ratio.append(edge_peak / edge_budget)
            self._window_cloud_ratio.append(cloud_peak / cloud_budget)

        if self.n_calls % self.log_freq == 0 and len(self._window_feasible) >= self.log_freq:
            feasible_rate = float(np.mean(self._window_feasible[-self.log_freq:]))
            oom_rate = float(np.mean(self._window_oom[-self.log_freq:]))
            edge_ratio = float(np.mean(self._window_edge_ratio[-self.log_freq:]))
            cloud_ratio = float(np.mean(self._window_cloud_ratio[-self.log_freq:]))

            self.logger.record("feasibility/feasible_rate", feasible_rate)
            self.logger.record("feasibility/oom_rate", oom_rate)
            self.logger.record("feasibility/avg_edge_mem_ratio", edge_ratio)
            self.logger.record("feasibility/avg_cloud_mem_ratio", cloud_ratio)

            print(
                f"[{self.num_timesteps} timesteps] "
                f"feasible_rate={feasible_rate:.3f}, "
                f"oom_rate={oom_rate:.3f}, "
                f"edge_ratio={edge_ratio:.3f}, "
                f"cloud_ratio={cloud_ratio:.3f}"
            )
        return True


class SyncEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        eval_freq: int,
        n_eval_episodes: int,
        best_model_save_path: str,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.best_mean_reward = -np.inf

        os.makedirs(best_model_save_path, exist_ok=True)

    def _sync_normalization(self):
        train_env = self.model.get_env()
        if isinstance(train_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
            self.eval_env.obs_rms = train_env.obs_rms
            self.eval_env.ret_rms = train_env.ret_rms

    def _save_best(self):
        model_path = os.path.join(self.best_model_save_path, "best_model")
        vecnorm_path = os.path.join(self.best_model_save_path, "vec_normalize.pkl")

        self.model.save(model_path)
        self.eval_env.save(vecnorm_path)

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        self._sync_normalization()
        self.eval_env.training = False
        self.eval_env.norm_reward = False

        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True,
            return_episode_rewards=False,
        )

        self.logger.record("eval/mean_reward", float(mean_reward))
        self.logger.record("eval/std_reward", float(std_reward))

        if self.verbose > 0:
            print(
                f"[Eval @ {self.num_timesteps} steps] "
                f"mean_reward={mean_reward:.4f}, std_reward={std_reward:.4f}, "
                f"best={self.best_mean_reward:.4f}"
            )

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self._save_best()
            if self.verbose > 0:
                print(f"✅ New best model saved. mean_reward={mean_reward:.4f}")

        return True


def make_env(rank: int, seed: int, config_dir: str = "configs") -> Callable:
    def _init():
        env = DeepFlowEnv(
            config_dir=config_dir,
            total_batch_size=32,
            episode_len=1,
            domain_randomization=True,
            seed=seed + rank,
            reward_mode="shaped",
        )
        env = Monitor(env)
        return env
    return _init


def make_eval_env(config_dir: str = "configs"):
    def _init():
        env = DeepFlowEnv(
            config_dir=config_dir,
            total_batch_size=32,
            episode_len=1,
            domain_randomization=True,
            seed=12345,
            reward_mode="shaped",
        )
        env = Monitor(env)
        return env

    env = DummyVecEnv([_init])
    env = VecMonitor(env)
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
    )
    return env


def train():
    print("=" * 78)
    print("🚀 DeepFlow-RL - PPO Retraining with Updated Reward Shaping")
    print("=" * 78)

    models_dir = "models/ppo_deepflow"
    best_dir = os.path.join(models_dir, "best_model")
    ckpt_dir = os.path.join(models_dir, "checkpoints")
    log_dir = "logs/ppo_memory_v3_rewardfix"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    seed = 42
    num_envs = 8

    env_fns = [make_env(rank=i, seed=seed, config_dir="configs") for i in range(num_envs)]
    train_env = SubprocVecEnv(env_fns)
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
        training=True,
    )

    eval_env = make_eval_env(config_dir="configs")

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        learning_rate=2e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.03,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_dir,
        seed=seed,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(5000 // num_envs, 1),
        save_path=ckpt_dir,
        name_prefix="ppo_memory_v3_rewardfix",
        save_vecnormalize=True,
    )

    sync_eval_callback = SyncEvalCallback(
        eval_env=eval_env,
        eval_freq=max(10000 // num_envs, 1),
        n_eval_episodes=50,
        best_model_save_path=best_dir,
        verbose=1,
    )

    strategy_logger = StrategyLoggerCallback(log_freq=2000)
    feasibility_logger = FeasibilityLoggerCallback(log_freq=2000)

    callback = CallbackList([
        checkpoint_callback,
        sync_eval_callback,
        strategy_logger,
        feasibility_logger,
    ])

    total_timesteps = 350_000

    print(f"Training with {num_envs} parallel envs, total_timesteps={total_timesteps}")
    print("TensorBoard logdir:", log_dir)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
        tb_log_name="ppo_memory_v3_rewardfix",
    )

    print("\n✅ Training finished. Saving final model...")

    final_model_path = os.path.join(models_dir, "final_model")
    final_vecnorm_path = os.path.join(models_dir, "vec_normalize.pkl")

    model.save(final_model_path)
    train_env.save(final_vecnorm_path)

    print(f"Saved final model to {final_model_path}.zip")
    print(f"Saved normalization stats to {final_vecnorm_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    train()