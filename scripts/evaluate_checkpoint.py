"""
Evaluate a saved PPO checkpoint on an arbitrary date window.

Usage:
    python scripts/evaluate_checkpoint.py \
        --checkpoint checkpoints/best_combined_5m/best/best_model.zip \
        --eval-start 2020-01-01 --eval-end 2020-12-31 \
        --label "2020 COVID" --n-episodes 30

    # Quick multi-period sweep (comma-separated):
    python scripts/evaluate_checkpoint.py \
        --checkpoint checkpoints/best_combined_5m/best/best_model.zip \
        --periods "2019-01-01:2019-12-31:2019 Val,2020-01-01:2020-12-31:2020 COVID,2022-01-01:2022-12-31:2022 Fed"
"""
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyarrow.parquet  # noqa: F401  — must precede torch/SB3

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from sb3_contrib import RecurrentPPO
    _RPPO_AVAILABLE = True
except ImportError:
    _RPPO_AVAILABLE = False


def _load_model(path: str):
    """Try PPO first, then SAC, then RecurrentPPO."""
    try:
        return PPO.load(path)
    except Exception:
        pass
    try:
        return SAC.load(path)
    except Exception:
        pass
    if _RPPO_AVAILABLE:
        return RecurrentPPO.load(path)
    raise ValueError(f"Could not load checkpoint as PPO, SAC, or RecurrentPPO: {path}")


def _is_recurrent(model) -> bool:
    return _RPPO_AVAILABLE and isinstance(model, RecurrentPPO)

from src.utils.config_loader import load_config
from src.utils.logging_config import setup_logging
from src.environment.data_store import MarketDataStore
from src.environment.trading_env import TradingEnv
from src.environment.wrappers import FlattenDictObservation


def make_eval_env(data_store, cfg, eval_start, eval_end, seed=0):
    def _init():
        env = TradingEnv(
            data_store,
            start_date=eval_start,
            end_date=eval_end,
            lookback=cfg["features"]["lookback_window"],
            episode_length=cfg["environment"]["episode_length"],
            initial_capital=cfg["environment"]["initial_capital"],
            transaction_cost_bps=cfg["environment"]["transaction_cost_bps"],
            reward_alpha=cfg["reward"]["excess_return_weight"],
            reward_beta=cfg["reward"]["drawdown_penalty_weight"],
            reward_gamma=cfg["reward"]["transaction_cost_weight"],
            drawdown_threshold=cfg["reward"]["drawdown_threshold"],
            regime_weights=None,   # uniform sampling for eval
            synthetic_store=None,  # no synthetic during eval
            synthetic_ratio=0.0,
            seed=seed,
        )
        return FlattenDictObservation(env)
    return _init


def evaluate_period(model, data_store, cfg, eval_start, eval_end, n_episodes, seed):
    vec_env = DummyVecEnv([make_eval_env(data_store, cfg, eval_start, eval_end, seed)])

    episode_rewards = []
    episode_returns = []
    episode_maxdds  = []

    obs          = vec_env.reset()
    ep_reward    = 0.0
    ep_cum_ret   = 0.0
    ep_maxdd     = 0.0
    done_count   = 0
    ep_start_nav = None

    # RPPO requires carrying LSTM state between steps and resetting at episode end
    recurrent    = _is_recurrent(model)
    lstm_state   = None
    ep_start_flag = np.ones((1,), dtype=bool)   # True = start of episode

    max_steps = n_episodes * cfg["environment"]["episode_length"] * 3

    for _ in range(max_steps):
        if recurrent:
            action, lstm_state = model.predict(
                obs, state=lstm_state, episode_start=ep_start_flag, deterministic=True
            )
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, info = vec_env.step(action)
        ep_start_flag = done   # reset LSTM at episode boundaries

        ep_reward += float(reward[0])
        nav = float(info[0].get("nav", 0.0))
        if ep_start_nav is None:
            ep_start_nav = nav
        net_ret = float(info[0].get("net_return", info[0].get("portfolio_return", 0.0)))
        ep_cum_ret += net_ret
        dd = float(info[0].get("drawdown", 0.0))
        if dd > ep_maxdd:
            ep_maxdd = dd

        if done[0]:
            if ep_start_nav and ep_start_nav > 0:
                total_return = (nav / ep_start_nav) - 1.0
            else:
                total_return = ep_cum_ret
            episode_rewards.append(ep_reward)
            episode_returns.append(total_return)
            episode_maxdds.append(ep_maxdd)
            done_count   += 1
            ep_reward     = 0.0
            ep_cum_ret    = 0.0
            ep_maxdd      = 0.0
            ep_start_nav  = None
            if done_count >= n_episodes:
                break

    vec_env.close()

    if not episode_rewards:
        return None

    returns = np.array(episode_returns)
    rewards = np.array(episode_rewards)
    maxdds  = np.array(episode_maxdds)

    # Annualised Sharpe: mean/std of per-episode returns, scaled by sqrt(252/ep_len)
    ep_len  = cfg["environment"]["episode_length"]
    ann_factor = np.sqrt(252 / ep_len)
    sharpe = (returns.mean() / (returns.std() + 1e-8)) * ann_factor

    return {
        "n_episodes":    done_count,
        "mean_reward":   float(rewards.mean()),
        "std_reward":    float(rewards.std()),
        "mean_return":   float(returns.mean()),
        "std_return":    float(returns.std()),
        "median_return": float(np.median(returns)),
        "win_rate":      float((returns > 0).mean()),
        "mean_maxdd":    float(maxdds.mean()),
        "sharpe_proxy":  float(sharpe),
    }


def print_row(label, r):
    if r is None:
        print(f"  {label:<22}  [no episodes completed]")
        return
    print(
        f"  {label:<22}  "
        f"reward {r['mean_reward']:+.4f}±{r['std_reward']:.4f}  "
        f"return {r['mean_return']*100:+.2f}%  "
        f"win {r['win_rate']*100:.0f}%  "
        f"maxDD {r['mean_maxdd']*100:.1f}%  "
        f"sharpe {r['sharpe_proxy']:+.2f}  "
        f"(n={r['n_episodes']})"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to .zip checkpoint file")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--eval-start", default=None)
    p.add_argument("--eval-end",   default=None)
    p.add_argument("--label",      default="eval")
    p.add_argument("--n-episodes", type=int, default=30)
    p.add_argument("--periods",    default=None,
                   help="Comma-separated 'start:end:label' triples for multi-period sweep")
    p.add_argument("--seed",       type=int, default=42)
    # Override episode_length (e.g. the best-combined config used 126)
    p.add_argument("--episode-length", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()

    cfg = load_config(args.config)
    if args.episode_length is not None:
        cfg["environment"]["episode_length"] = args.episode_length

    print(f"Loading checkpoint: {args.checkpoint}")
    model = _load_model(args.checkpoint)

    print("Loading MarketDataStore...")
    data_store = MarketDataStore(config_path=args.config)

    periods = []
    if args.periods:
        for tok in args.periods.split(","):
            parts = tok.strip().split(":")
            periods.append((parts[0], parts[1], parts[2] if len(parts) > 2 else f"{parts[0]}–{parts[1]}"))
    elif args.eval_start and args.eval_end:
        periods.append((args.eval_start, args.eval_end, args.label))
    else:
        # Default: val + two bear test years
        periods = [
            ("2019-01-01", "2019-12-31", "2019 Val"),
            ("2020-01-01", "2020-12-31", "2020 COVID"),
            ("2022-01-01", "2022-12-31", "2022 Fed"),
        ]

    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Episodes per period: {args.n_episodes}  |  ep_len={cfg['environment']['episode_length']}")
    print("-" * 90)
    for eval_start, eval_end, label in periods:
        r = evaluate_period(model, data_store, cfg, eval_start, eval_end,
                            args.n_episodes, args.seed)
        print_row(label, r)
    print("-" * 90)


if __name__ == "__main__":
    main()
