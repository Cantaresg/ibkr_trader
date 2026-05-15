"""
IntradayTrainer: wires IntradayTradingEnv + FlattenDictObservation to SB3 PPO
or RecurrentPPO (RPPO) with IntradayFeaturesExtractor / MlpLstmPolicy.

Reads the flat intraday_trader/config.yaml structure.

algo: "ppo"  — standard PPO (MlpPolicy + IntradayFeaturesExtractor)
      "rppo" — RecurrentPPO with LSTM (MlpLstmPolicy, requires sb3-contrib)
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import pyarrow.parquet  # noqa: F401  — must precede torch/SB3 on Windows

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

try:
    from sb3_contrib import RecurrentPPO
    _RPPO_AVAILABLE = True
except ImportError:
    _RPPO_AVAILABLE = False

from src.environment.wrappers import FlattenDictObservation
from intraday_trader.constants import LOOKBACK, N_STOCKS
from intraday_trader.data_store import IntradayDataStore
from intraday_trader.env import IntradayTradingEnv
from intraday_trader.mlp_policy import IntradayFeaturesExtractor
from src.training.callbacks import EpisodeMetricsCallback, SharpeEvalCallback
from src.utils.logging_config import get_logger
from src.utils.seed import set_global_seed

log = get_logger("intraday.trainer")


def _make_env(data_store: IntradayDataStore, cfg: dict, seed: int, syn_store=None):
    """Factory for a single IntradayTradingEnv wrapped in FlattenDictObservation."""
    def _init():
        env = IntradayTradingEnv(
            data_store,
            start_date           = cfg["training_start"],
            end_date             = cfg["training_end"],
            lookback             = cfg["lookback"],
            n_stocks             = cfg.get("n_stocks", N_STOCKS),
            n_days_per_episode   = cfg["n_days_per_episode"],
            initial_capital      = cfg["initial_capital"],
            transaction_cost_bps = cfg["transaction_cost_bps"],
            min_position_weight  = cfg.get("min_position_weight", 0.0),
            reward_alpha         = cfg["reward_alpha"],
            reward_beta          = cfg["reward_beta"],
            reward_gamma         = cfg["reward_gamma"],
            reward_delta         = cfg["reward_delta"],
            reward_zeta          = cfg.get("reward_zeta", 0.0),
            reward_eta           = cfg.get("reward_eta", 0.0),
            drawdown_threshold   = cfg["drawdown_threshold"],
            seed                 = seed,
            synthetic_store      = syn_store,
            synthetic_ratio      = cfg.get("synthetic_ratio", 0.0),
            intraday_scanner_enabled      = cfg.get("intraday_scanner_enabled", False),
            intraday_refresh_every_n_bars = cfg.get("intraday_refresh_every_n_bars", 1),
            regime_balanced_sampling      = cfg.get("regime_balanced_sampling", False),
            eod_force_flat                = cfg.get("eod_force_flat", True),
        )
        return FlattenDictObservation(env)
    return _init


class IntradayTrainer:
    """
    Wraps model setup, training loop, checkpointing, and evaluation
    for the intraday DRL system.

    algo: "ppo"  — standard PPO (MlpPolicy + IntradayFeaturesExtractor)
            "rppo" — RecurrentPPO with LSTM (requires sb3-contrib)
    """

    def __init__(
        self,
        config: dict,
        config_path: str = "intraday_trader/config.yaml",
        run_name: str = "intraday_ppo",
        warm_start_path: Optional[str] = None,
        algo: str = "ppo",
        synthetic_ratio: float | None = None,
        synthetic_dir: str | None = None,
    ):
        self.cfg      = config
        self.run_name = run_name
        self.algo     = algo.lower()

        if self.algo == "rppo" and not _RPPO_AVAILABLE:
            raise ImportError(
                "RecurrentPPO requires sb3-contrib: pip install sb3-contrib"
            )

        # Resolve algo-specific config block (fall back to ppo if section missing)
        algo_key = self.algo if self.algo in config else "ppo"
        algo_cfg   = config[algo_key]
        env_cfg    = config["environment"]
        reward_cfg = config["reward"]
        train_cfg  = config["training"]
        data_cfg   = config["data"]
        feat_cfg   = config["features"]
        syn_cfg    = config.get("synthetic", {})
        scanner_cfg = config.get("scanner", {})
        intraday_scanner_cfg = scanner_cfg.get("intraday", {})
        # keep ppo_cfg name for backward compat with rest of method
        ppo_cfg    = algo_cfg

        seed = config.get("project", {}).get("seed", 42)
        set_global_seed(seed)

        # Synthetic ratio / dir may be overridden via CLI args
        _syn_ratio = synthetic_ratio if synthetic_ratio is not None else syn_cfg.get("synthetic_ratio", 0.0)
        _syn_dir   = synthetic_dir   or syn_cfg.get("synthetic_dir", None)

        _n_stocks = config.get("universe", {}).get("n_stocks", N_STOCKS)
        self._flat_cfg = {
            "training_start":       data_cfg.get("start_date",          "2022-01-01"),
            "training_end":         data_cfg.get("train_end",            "2024-06-30"),
            "lookback":             feat_cfg.get("lookback_bars",        LOOKBACK),
            "n_stocks":             _n_stocks,
            "n_days_per_episode":   env_cfg["n_days_per_episode"],
            "initial_capital":      env_cfg["initial_capital"],
            "transaction_cost_bps": env_cfg["transaction_cost_bps"],
            "min_position_weight":  env_cfg.get("min_position_weight", 0.0),
            "reward_alpha":         reward_cfg["excess_return_weight"],
            "reward_beta":          reward_cfg["drawdown_penalty_weight"],
            "reward_gamma":         reward_cfg["transaction_cost_weight"],
            "reward_delta":         reward_cfg["overnight_exposure_weight"],
            "reward_zeta":          reward_cfg.get("diversification_weight", 0.0),
            "reward_eta":           reward_cfg.get("sortino_asymmetry_weight", 0.0),
            "drawdown_threshold":   reward_cfg["drawdown_threshold"],
            "synthetic_ratio":      _syn_ratio,
            "intraday_scanner_enabled": intraday_scanner_cfg.get("enabled", False),
            "intraday_refresh_every_n_bars": intraday_scanner_cfg.get("refresh_every_n_bars", 1),
            "regime_balanced_sampling": env_cfg.get("regime_balanced_sampling", False),
            "eod_force_flat":           env_cfg.get("eod_force_flat", True),
        }

        log.info("Loading IntradayDataStore from %s...", config_path)
        self.data_store = IntradayDataStore(config_path=config_path)

        # Load synthetic store if directory provided and ratio > 0
        self._syn_store = None
        if _syn_dir and _syn_ratio > 0.0:
            from intraday_trader.synthetic_store import IntradaySyntheticStore
            self._syn_store = IntradaySyntheticStore(_syn_dir)
            if self._syn_store.is_empty():
                log.warning(
                    "Synthetic store is empty at %s — disabling synthetic augmentation. "
                    "Run generate_synthetic.py first.",
                    _syn_dir,
                )
                self._syn_store = None
            else:
                log.info(
                    "Synthetic store loaded: %d episodes (ratio=%.0f%%)",
                    len(self._syn_store), _syn_ratio * 100,
                )

        n_envs     = ppo_cfg["n_envs"]
        device_str = ppo_cfg.get("device", "cuda")
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        if device_str == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA requested but not available — falling back to CPU")

        log.info("Creating %d training envs (algo=%s)...", n_envs, self.algo)
        env_fns = [
            _make_env(self.data_store, self._flat_cfg, seed + i, syn_store=self._syn_store)
            for i in range(n_envs)
        ]
        self.vec_env = DummyVecEnv(env_fns)

        eval_flat_cfg = {
            **self._flat_cfg,
            "training_start":  data_cfg.get("eval_start", "2024-07-01"),
            "training_end":    data_cfg.get("eval_end",   "2024-12-31"),
            "synthetic_ratio": 0.0,   # no synthetic during eval
        }
        self.eval_env = DummyVecEnv([_make_env(self.data_store, eval_flat_cfg, seed + 999)])

        features_dim = 256
        # Base policy kwargs — used as-is for PPO; RPPO extends with LSTM params
        self._base_policy_kwargs = dict(
            features_extractor_class=IntradayFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=features_dim, n_stocks=_n_stocks),
            net_arch=[features_dim, features_dim // 2],
            activation_fn=nn.ReLU,
        )
        # Legacy attribute kept for any external references
        self._policy_kwargs = self._base_policy_kwargs

        ckpt_dir = Path(train_cfg["checkpoint_dir"]) / run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._ckpt_dir  = ckpt_dir
        self._train_cfg = train_cfg
        self._ppo_cfg   = ppo_cfg

        self.model = self._build_model(
            warm_start_path,
            str(train_cfg["tensorboard_log"]),
            seed,
        )

    # ------------------------------------------------------------------
    def _build_model(self, warm_start_path: Optional[str], tb_log: str, seed: int):
        """Build (or warm-start) a PPO or RecurrentPPO model."""
        cfg = self._ppo_cfg
        dev = str(self.device)

        if self.algo == "rppo":
            ModelClass  = RecurrentPPO
            policy_name = "MlpLstmPolicy"
            lstm_pkw = {
                **self._base_policy_kwargs,
                "lstm_hidden_size":   cfg.get("lstm_hidden_size",  128),
                "n_lstm_layers":      cfg.get("n_lstm_layers",     1),
                "shared_lstm":        False,
                "enable_critic_lstm": True,
            }
            policy_kwargs = lstm_pkw
        else:
            ModelClass    = PPO
            policy_name   = "MlpPolicy"
            policy_kwargs = self._base_policy_kwargs

        common_kwargs = dict(
            learning_rate   = cfg["learning_rate"],
            n_steps         = cfg["n_steps"],
            batch_size      = cfg["batch_size"],
            n_epochs        = cfg["n_epochs"],
            gamma           = cfg["gamma"],
            gae_lambda      = cfg["gae_lambda"],
            clip_range      = cfg["clip_range"],
            ent_coef        = cfg["ent_coef"],
            vf_coef         = cfg["vf_coef"],
            max_grad_norm   = cfg["max_grad_norm"],
            target_kl       = cfg.get("target_kl"),
            tensorboard_log = tb_log,
            device          = dev,
            verbose         = 1,
            seed            = seed,
        )
        load_kwargs = dict(
            learning_rate = cfg["learning_rate"],
            ent_coef      = cfg["ent_coef"],
            clip_range    = cfg["clip_range"],
            target_kl     = cfg.get("target_kl"),
            max_grad_norm = cfg["max_grad_norm"],
        )

        if warm_start_path and Path(warm_start_path).exists():
            log.info("Warm-starting %s from %s", self.algo.upper(), warm_start_path)
            return ModelClass.load(
                warm_start_path,
                env=self.vec_env,
                device=dev,
                tensorboard_log=tb_log,
                **load_kwargs,
            )

        log.info("Building new %s model for intraday", self.algo.upper())
        return ModelClass(
            policy_name,
            env=self.vec_env,
            policy_kwargs=policy_kwargs,
            **common_kwargs,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _latest_checkpoint(ckpt_dir: Path, algo: str) -> tuple[Path | None, int]:
        """Return (path, step_count) of the most recent checkpoint zip, or (None, 0)."""
        prefix = f"intraday_{algo}_"
        zips = sorted(
            (p for p in ckpt_dir.glob(f"{prefix}*_steps.zip")),
            key=lambda p: int(p.stem.split("_steps")[0].rsplit("_", 1)[-1]),
        )
        if not zips:
            return None, 0
        latest = zips[-1]
        step = int(latest.stem.split("_steps")[0].rsplit("_", 1)[-1])
        return latest, step

    def train(self, total_timesteps: Optional[int] = None, resume: bool = False) -> None:
        steps  = total_timesteps or self._ppo_cfg["total_timesteps"]
        n_envs = self._ppo_cfg["n_envs"]
        reset_num_timesteps = True

        if resume:
            ckpt_path, resumed_steps = self._latest_checkpoint(self._ckpt_dir, self.algo)
            if ckpt_path:
                log.info("Resuming %s from %s  (%d steps done, %d remaining)",
                         self.algo.upper(), ckpt_path, resumed_steps, steps - resumed_steps)
                try:
                    from sb3_contrib import RecurrentPPO as _RPPO
                except ImportError:
                    _RPPO = None
                from stable_baselines3 import PPO as _PPO
                ModelClass = _RPPO if (self.algo == "rppo" and _RPPO) else _PPO
                dev = str(self.device)
                cfg = self._ppo_cfg
                self.model = ModelClass.load(
                    str(ckpt_path),
                    env=self.vec_env,
                    device=dev,
                    tensorboard_log=str(self._train_cfg["tensorboard_log"]),
                    learning_rate=cfg["learning_rate"],
                    ent_coef=cfg["ent_coef"],
                    clip_range=cfg["clip_range"],
                    target_kl=cfg.get("target_kl"),
                    max_grad_norm=cfg["max_grad_norm"],
                )
                steps = max(steps - resumed_steps, 0)
                reset_num_timesteps = False
            else:
                log.warning("--resume requested but no checkpoint found in %s — starting fresh", self._ckpt_dir)

        log.info("Starting intraday %s training: %d timesteps  device=%s",
                 self.algo.upper(), steps, self.device)

        ckpt_freq = self._train_cfg["checkpoint_freq"] // n_envs
        eval_freq = self._train_cfg["eval_freq"]       // n_envs

        checkpoint_cb = CheckpointCallback(
            save_freq=ckpt_freq,
            save_path=str(self._ckpt_dir),
            name_prefix=f"intraday_{self.algo}",
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
        log.info("Intraday training complete. Model saved to %s", final_path)

    # ------------------------------------------------------------------
    def evaluate(self, n_episodes: int = 10, deterministic: bool = True) -> dict:
        """
        Evaluate the trained policy using IntradayPolicyRunner to produce
        both SB3 mean/std reward and financial metrics (daily Sharpe, max
        drawdown, win rate, annualised return).
        """
        from stable_baselines3.common.evaluation import evaluate_policy
        from intraday_trader.backtester import IntradayPolicyRunner

        # --- SB3 reward stats (fast, uses the already-wrapped eval_env) ---
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=n_episodes,
            deterministic=deterministic,
        )
        log.info("Intraday eval reward over %d episodes: %.4f ± %.4f",
                 n_episodes, mean_reward, std_reward)

        # --- Financial metrics via IntradayPolicyRunner ---
        # Save the model to a temp path so PolicyRunner can load it
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            self.model.save(tmp_path.replace(".zip", ""))
            zip_path = tmp_path.replace(".zip", "") + ".zip"
            if not os.path.exists(zip_path):
                zip_path = tmp_path  # fallback: some SB3 versions omit the extension

            data_cfg = self.cfg.get("data", {})
            eval_start = data_cfg.get("eval_start", "2024-07-01")
            eval_end   = data_cfg.get("eval_end",   "2024-12-31")
            runner = IntradayPolicyRunner(
                model_path          = zip_path,
                data_store          = self.data_store,
                start_date          = eval_start,
                end_date            = eval_end,
                deterministic       = deterministic,
                seed                = self.cfg.get("project", {}).get("seed", 42),
                min_position_weight = self._flat_cfg.get("min_position_weight", 0.0),
                eod_force_flat      = self._flat_cfg.get("eod_force_flat", True),
            )
            fin_result = runner.run(n_episodes=n_episodes)
            fin_metrics = fin_result["mean_metrics"]
        except Exception as exc:
            log.warning("Financial metric computation failed: %s — returning reward stats only", exc)
            fin_metrics = {}
        finally:
            for p in [tmp_path, tmp_path.replace(".zip", "") + ".zip",
                      tmp_path.replace(".zip", "")]:
                try:
                    os.remove(p)
                except OSError:
                    pass

        result = {
            "mean_reward":    mean_reward,
            "std_reward":     std_reward,
            **fin_metrics,
        }

        if fin_metrics:
            log.info(
                "Financial metrics: DailySharpe=%.3f  BarSharpe=%.3f  MDD=%.1f%%  WinRate=%.1f%%  AnnRet=%.1f%%",
                fin_metrics.get("daily_sharpe", 0.0),
                fin_metrics.get("bar_sharpe", 0.0),
                fin_metrics.get("max_drawdown", 0.0) * 100,
                fin_metrics.get("daily_win_rate", 0.0) * 100,
                fin_metrics.get("annualised_return", 0.0) * 100,
            )

        return result
