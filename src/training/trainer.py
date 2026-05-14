"""
Multi-algorithm trainer for TradingEnv.

Supports PPO (on-policy), SAC (off-policy), and RecurrentPPO (PPO + LSTM).

Usage (via scripts/train_agent.py):
    trainer = Trainer(cfg, algo="ppo")
    trainer.train()
    trainer.evaluate()
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

import pyarrow.parquet  # noqa: F401  — must precede torch/SB3 on Windows

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

try:
    from sb3_contrib import RecurrentPPO
    _RPPO_AVAILABLE = True
except ImportError:
    _RPPO_AVAILABLE = False

from src.environment.data_store import MarketDataStore
from src.environment.trading_env import TradingEnv
from src.environment.wrappers import FlattenDictObservation
from src.environment.synthetic_store import SyntheticEpisodeStore
from src.models.mlp_policy import TradingFeaturesExtractor
from src.training.callbacks import EpisodeMetricsCallback, SharpeEvalCallback
from src.utils.logging_config import get_logger
from src.utils.seed import set_global_seed

log = get_logger("training.trainer")


def _make_env(data_store: MarketDataStore, cfg: dict, seed: int, synthetic_store=None):
    """Factory for a single TradingEnv wrapped in FlattenDictObservation."""
    def _init():
        env = TradingEnv(
            data_store,
            start_date=cfg["training_start"],
            end_date=cfg["training_end"],
            lookback=cfg["lookback"],
            episode_length=cfg["episode_length"],
            initial_capital=cfg["initial_capital"],
            transaction_cost_bps=cfg["transaction_cost_bps"],
            reward_alpha=cfg["reward_alpha"],
            reward_beta=cfg["reward_beta"],
            reward_gamma=cfg["reward_gamma"],
            drawdown_threshold=cfg["drawdown_threshold"],
            regime_weights=cfg.get("regime_weights"),
            synthetic_store=synthetic_store,
            synthetic_ratio=cfg.get("synthetic_ratio", 0.0),
            seed=seed,
        )
        return FlattenDictObservation(env)
    return _init


class Trainer:
    """
    Wraps model setup, training loop, checkpointing, and evaluation.
    algo: "ppo" | "sac" | "rppo"
    """

    def __init__(
        self,
        config: dict,
        run_name: str = "phase1_mlp",
        warm_start_path: Optional[str] = None,
        algo: str = "ppo",
    ):
        self.cfg      = config
        self.run_name = run_name
        self.algo     = algo.lower()

        if self.algo == "rppo" and not _RPPO_AVAILABLE:
            raise ImportError(
                "RecurrentPPO requires sb3-contrib: pip install sb3-contrib"
            )

        seed = config.get("seed", 42)
        set_global_seed(seed)

        # Resolve algo-specific config block
        algo_key = "rppo" if self.algo == "rppo" else self.algo
        if algo_key not in config:
            log.warning("No '%s' section in config — falling back to ppo config", algo_key)
            algo_key = "ppo"
        self._algo_cfg = config[algo_key]

        env_cfg    = config["environment"]
        data_cfg   = config["data"]
        train_cfg  = config["training"]
        reward_cfg = config["reward"]

        self._flat_cfg = {
            "training_start":       config.get("training_start", "2013-01-01"),
            "training_end":         config.get("training_end",   "2018-12-31"),
            "lookback":             config["features"]["lookback_window"],
            "episode_length":       env_cfg["episode_length"],
            "initial_capital":      env_cfg["initial_capital"],
            "transaction_cost_bps": env_cfg["transaction_cost_bps"],
            "reward_alpha":         reward_cfg["excess_return_weight"],
            "reward_beta":          reward_cfg["drawdown_penalty_weight"],
            "reward_gamma":         reward_cfg["transaction_cost_weight"],
            "drawdown_threshold":   reward_cfg["drawdown_threshold"],
            "regime_weights":       config.get("regime_weights"),
            "synthetic_ratio":      config.get("synthetic_ratio", 0.0),
        }

        log.info("Loading MarketDataStore...")
        self.data_store = MarketDataStore(
            config_path=config.get("config_path", "config/config.yaml"),
        )

        syn_dir   = config.get("synthetic_dir")
        syn_store = None
        if syn_dir and Path(syn_dir).exists():
            log.info("Loading SyntheticEpisodeStore from %s...", syn_dir)
            syn_store = SyntheticEpisodeStore(syn_dir)
        elif syn_dir:
            log.warning("synthetic_dir '%s' not found — training without synthetic episodes", syn_dir)

        n_envs     = self._algo_cfg["n_envs"]
        device_str = self._algo_cfg.get("device", "cuda")
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        if device_str == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA requested but not available — falling back to CPU")

        log.info("Creating %d training envs (algo=%s, synthetic_ratio=%.2f)...",
                 n_envs, self.algo, self._flat_cfg["synthetic_ratio"])
        env_fns = [_make_env(self.data_store, self._flat_cfg, seed + i, syn_store)
                   for i in range(n_envs)]
        self.vec_env = DummyVecEnv(env_fns)

        eval_cfg = {**self._flat_cfg,
                    "training_start": config.get("eval_start", "2019-01-01"),
                    "training_end":   config.get("eval_end",   "2019-12-31"),
                    "synthetic_ratio": 0.0}
        self.eval_env = DummyVecEnv([_make_env(self.data_store, eval_cfg, seed + 999)])

        features_dim = config.get("model", {}).get("features_dim", 512)
        self._base_policy_kwargs = dict(
            features_extractor_class=TradingFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=features_dim),
            net_arch=[features_dim, features_dim // 2],
            activation_fn=nn.ReLU,
        )

        ckpt_dir = Path(train_cfg["checkpoint_dir"]) / run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        self._ckpt_dir  = ckpt_dir
        self._train_cfg = train_cfg
        self.model      = self._build_model(warm_start_path, str(train_cfg["tensorboard_log"]), seed)

    # ------------------------------------------------------------------
    def _build_model(self, warm_start_path: Optional[str], tb_log: str, seed: int):
        cfg   = self._algo_cfg
        dev   = str(self.device)
        pkw   = self._base_policy_kwargs

        if self.algo == "sac":
            ModelClass  = SAC
            policy_name = "MlpPolicy"
            init_kwargs = dict(
                learning_rate=cfg["learning_rate"],
                buffer_size=cfg["buffer_size"],
                batch_size=cfg["batch_size"],
                tau=cfg["tau"],
                gamma=cfg["gamma"],
                train_freq=cfg["train_freq"],
                gradient_steps=cfg["gradient_steps"],
                learning_starts=cfg["learning_starts"],
                ent_coef=cfg.get("ent_coef", "auto"),
                policy_kwargs=pkw,
                tensorboard_log=tb_log,
                device=dev,
                verbose=1,
            )
            load_kwargs = dict(
                learning_rate=cfg["learning_rate"],
                ent_coef=cfg.get("ent_coef", "auto"),
                tau=cfg["tau"],
            )

        elif self.algo == "rppo":
            ModelClass  = RecurrentPPO
            policy_name = "MlpLstmPolicy"
            lstm_pkw = {
                **pkw,
                "lstm_hidden_size":   cfg.get("lstm_hidden_size", 256),
                "n_lstm_layers":      cfg.get("n_lstm_layers", 1),
                "shared_lstm":        False,
                "enable_critic_lstm": True,
            }
            init_kwargs = dict(
                learning_rate=cfg["learning_rate"],
                n_steps=cfg["n_steps"],
                batch_size=cfg["batch_size"],
                n_epochs=cfg["n_epochs"],
                gamma=cfg["gamma"],
                gae_lambda=cfg["gae_lambda"],
                clip_range=cfg["clip_range"],
                ent_coef=cfg["ent_coef"],
                vf_coef=cfg["vf_coef"],
                max_grad_norm=cfg["max_grad_norm"],
                target_kl=cfg.get("target_kl"),
                policy_kwargs=lstm_pkw,
                tensorboard_log=tb_log,
                device=dev,
                verbose=1,
                seed=seed,
            )
            load_kwargs = dict(
                learning_rate=cfg["learning_rate"],
                ent_coef=cfg["ent_coef"],
                clip_range=cfg["clip_range"],
                target_kl=cfg.get("target_kl"),
                max_grad_norm=cfg["max_grad_norm"],
            )

        else:  # ppo (default)
            ModelClass  = PPO
            policy_name = "MlpPolicy"
            init_kwargs = dict(
                learning_rate=cfg["learning_rate"],
                n_steps=cfg["n_steps"],
                batch_size=cfg["batch_size"],
                n_epochs=cfg["n_epochs"],
                gamma=cfg["gamma"],
                gae_lambda=cfg["gae_lambda"],
                clip_range=cfg["clip_range"],
                ent_coef=cfg["ent_coef"],
                vf_coef=cfg["vf_coef"],
                max_grad_norm=cfg["max_grad_norm"],
                target_kl=cfg.get("target_kl"),
                policy_kwargs=pkw,
                tensorboard_log=tb_log,
                device=dev,
                verbose=1,
                seed=seed,
            )
            load_kwargs = dict(
                learning_rate=cfg["learning_rate"],
                ent_coef=cfg["ent_coef"],
                clip_range=cfg["clip_range"],
                target_kl=cfg.get("target_kl"),
                max_grad_norm=cfg["max_grad_norm"],
            )

        if warm_start_path and Path(warm_start_path).exists():
            log.info("Warm-starting from %s", warm_start_path)
            return ModelClass.load(
                warm_start_path,
                env=self.vec_env,
                device=dev,
                tensorboard_log=tb_log,
                **load_kwargs,
            )

        log.info("Building new %s model", self.algo.upper())
        return ModelClass(policy_name, env=self.vec_env, **init_kwargs)

    # ------------------------------------------------------------------
    def train(self, total_timesteps: Optional[int] = None, reset_num_timesteps: bool = True) -> None:
        steps  = total_timesteps or self._algo_cfg["total_timesteps"]
        n_envs = self._algo_cfg["n_envs"]

        log.info("Starting training: %d timesteps  algo=%s  device=%s  reset_num_timesteps=%s",
                 steps, self.algo, self.device, reset_num_timesteps)

        ckpt_freq = self._train_cfg["checkpoint_freq"] // n_envs
        eval_freq = self._train_cfg["eval_freq"]       // n_envs

        checkpoint_cb = CheckpointCallback(
            save_freq=ckpt_freq,
            save_path=str(self._ckpt_dir),
            name_prefix=self.algo,
            save_vecnormalize=False,
        )
        metrics_cb = EpisodeMetricsCallback(verbose=0)
        eval_cb = SharpeEvalCallback(
            eval_env=self.eval_env,
            n_eval_episodes=self._train_cfg["eval_episodes"],
            eval_freq=eval_freq,
            best_model_save_path=str(self._ckpt_dir / "best"),
            log_path=str(self._ckpt_dir / "eval_logs"),
            deterministic=True,
            verbose=1,
        )

        callbacks = CallbackList([checkpoint_cb, metrics_cb, eval_cb])

        self.model.learn(
            total_timesteps=steps,
            callback=callbacks,
            tb_log_name=self.run_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=True,
        )

        final_path = self._ckpt_dir / "final_model"
        self.model.save(str(final_path))
        log.info("Training complete. Final model saved to %s", final_path)

    # ------------------------------------------------------------------
    def evaluate(self, n_episodes: int = 20, deterministic: bool = True) -> dict:
        from stable_baselines3.common.evaluation import evaluate_policy
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=n_episodes,
            deterministic=deterministic,
        )
        log.info("Eval over %d episodes: mean_reward=%.4f ± %.4f",
                 n_episodes, mean_reward, std_reward)
        return {"mean_reward": mean_reward, "std_reward": std_reward}
