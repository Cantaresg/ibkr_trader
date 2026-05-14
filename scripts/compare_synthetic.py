"""
Compare synthetic-augmentation checkpoints against the no-synthetic baseline
on three out-of-sample periods: 2019 Val, 2020 COVID crash, 2022 Fed tightening.

Checkpoints evaluated:
  baseline     exp_22_dd_beta_03   dd_beta=0.3, ep_len=252, no synthetic
  syn_01       syn_01_ratio_25     dd_beta=0.3, ep_len=126, 25% synthetic
  syn_02       syn_02_ratio_50     dd_beta=0.3, ep_len=126, 50% synthetic
  syn_03       syn_03_ratio_75     dd_beta=0.3, ep_len=126, 75% synthetic
  best_5m      best_combined_5m    dd_beta=0.3, ep_len=126, 50% syn, 5M steps

Usage:
    python scripts/compare_synthetic.py
    python scripts/compare_synthetic.py --n-episodes 50
    python scripts/compare_synthetic.py --out results/synthetic_comparison.csv
"""
import sys
import argparse
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyarrow.parquet  # noqa: F401  — must precede torch/SB3
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.utils.config_loader import load_config
from src.utils.logging_config import setup_logging
from src.environment.data_store import MarketDataStore
from src.environment.trading_env import TradingEnv
from src.environment.wrappers import FlattenDictObservation


# ---------------------------------------------------------------------------
CHECKPOINTS = [
    {
        "label":    "baseline (no-syn, ep252)",
        "path":     "checkpoints/exp_22_dd_beta_03/best/best_model.zip",
        "ep_len":   252,
    },
    {
        "label":    "syn_01 (25%, ep126)",
        "path":     "checkpoints/syn_01_ratio_25/best/best_model.zip",
        "ep_len":   126,
    },
    {
        "label":    "syn_02 (50%, ep126)",
        "path":     "checkpoints/syn_02_ratio_50/best/best_model.zip",
        "ep_len":   126,
    },
    {
        "label":    "syn_03 (75%, ep126)",
        "path":     "checkpoints/syn_03_ratio_75/best/best_model.zip",
        "ep_len":   126,
    },
    {
        "label":    "best_5m (50%, ep126, 5M)",
        "path":     "checkpoints/best_combined_5m/best/best_model.zip",
        "ep_len":   126,
    },
]

PERIODS = [
    ("2019-01-01", "2019-12-31", "2019 Val"),
    ("2020-01-01", "2020-12-31", "2020 COVID"),
    ("2022-01-01", "2022-12-31", "2022 Fed"),
]


# ---------------------------------------------------------------------------
def make_eval_env(data_store, cfg, eval_start, eval_end, ep_len, seed=0):
    def _init():
        env = TradingEnv(
            data_store,
            start_date=eval_start,
            end_date=eval_end,
            lookback=cfg["features"]["lookback_window"],
            episode_length=ep_len,
            initial_capital=cfg["environment"]["initial_capital"],
            transaction_cost_bps=cfg["environment"]["transaction_cost_bps"],
            reward_alpha=cfg["reward"]["excess_return_weight"],
            reward_beta=cfg["reward"]["drawdown_penalty_weight"],
            reward_gamma=cfg["reward"]["transaction_cost_weight"],
            drawdown_threshold=cfg["reward"]["drawdown_threshold"],
            regime_weights=None,
            synthetic_store=None,
            synthetic_ratio=0.0,
            seed=seed,
        )
        return FlattenDictObservation(env)
    return _init


def evaluate_period(model, data_store, cfg, eval_start, eval_end, ep_len, n_episodes, seed):
    vec_env = DummyVecEnv([make_eval_env(data_store, cfg, eval_start, eval_end, ep_len, seed)])

    episode_rewards = []
    episode_returns = []
    episode_maxdds  = []
    done_count      = 0
    ep_reward       = 0.0
    ep_maxdd        = 0.0
    ep_start_nav    = None

    obs = vec_env.reset()
    max_steps = n_episodes * ep_len * 3

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        ep_reward += float(reward[0])
        nav        = float(info[0].get("nav", 0.0))
        if ep_start_nav is None:
            ep_start_nav = nav
        dd = float(info[0].get("drawdown", 0.0))
        if dd > ep_maxdd:
            ep_maxdd = dd

        if done[0]:
            total_return = (nav / ep_start_nav) - 1.0 if ep_start_nav and ep_start_nav > 0 else 0.0
            episode_rewards.append(ep_reward)
            episode_returns.append(total_return)
            episode_maxdds.append(ep_maxdd)
            done_count += 1
            ep_reward    = 0.0
            ep_maxdd     = 0.0
            ep_start_nav = None
            if done_count >= n_episodes:
                break

    vec_env.close()
    if not episode_rewards:
        return None

    returns     = np.array(episode_returns)
    rewards     = np.array(episode_rewards)
    maxdds      = np.array(episode_maxdds)
    ann_factor  = np.sqrt(252 / ep_len)
    sharpe      = (returns.mean() / (returns.std() + 1e-8)) * ann_factor

    return {
        "n":            done_count,
        "mean_reward":  float(rewards.mean()),
        "mean_return":  float(returns.mean()),
        "std_return":   float(returns.std()),
        "win_rate":     float((returns > 0).mean()),
        "mean_maxdd":   float(maxdds.mean()),
        "sharpe":       float(sharpe),
    }


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      default="config/config.yaml")
    p.add_argument("--n-episodes",  type=int, default=40)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--out",         default="results/synthetic_comparison.csv")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()
    cfg  = load_config(args.config)

    print("Loading MarketDataStore...")
    data_store = MarketDataStore(config_path=args.config)

    rows = []

    for ckpt in CHECKPOINTS:
        path = Path(ckpt["path"])
        if not path.exists():
            print(f"SKIP {ckpt['label']} — checkpoint not found: {path}")
            continue

        print(f"\nLoading {ckpt['label']}...")
        model = PPO.load(str(path))

        for eval_start, eval_end, period_label in PERIODS:
            print(f"  Evaluating {period_label}...", end=" ", flush=True)
            r = evaluate_period(
                model, data_store, cfg,
                eval_start, eval_end,
                ckpt["ep_len"],
                args.n_episodes,
                args.seed,
            )
            if r is None:
                print("no episodes")
                continue
            print(f"return={r['mean_return']*100:+.2f}%  sharpe={r['sharpe']:+.2f}  win={r['win_rate']*100:.0f}%  maxDD={r['mean_maxdd']*100:.1f}%")
            rows.append({
                "model":       ckpt["label"],
                "period":      period_label,
                "ep_len":      ckpt["ep_len"],
                "n":           r["n"],
                "return_pct":  round(r["mean_return"] * 100, 2),
                "std_pct":     round(r["std_return"]  * 100, 2),
                "win_rate":    round(r["win_rate"]     * 100, 1),
                "mean_maxdd":  round(r["mean_maxdd"]   * 100, 1),
                "sharpe":      round(r["sharpe"],             3),
                "mean_reward": round(r["mean_reward"],        4),
            })

    # -----------------------------------------------------------------------
    # Print comparison table
    print("\n" + "=" * 115)
    print(f"{'Model':<30}  {'Period':<14}  {'Return':>8}  {'StdDev':>7}  {'Win%':>5}  {'MaxDD%':>7}  {'Sharpe':>7}  {'Reward':>8}")
    print("=" * 115)
    last_model = None
    for row in rows:
        if row["model"] != last_model and last_model is not None:
            print("-" * 115)
        last_model = row["model"]
        print(
            f"  {row['model']:<28}  {row['period']:<14}  "
            f"{row['return_pct']:>+7.2f}%  {row['std_pct']:>6.2f}%  "
            f"{row['win_rate']:>5.0f}%  {row['mean_maxdd']:>6.1f}%  "
            f"{row['sharpe']:>+7.3f}  {row['mean_reward']:>+8.4f}"
        )
    print("=" * 115)

    # -----------------------------------------------------------------------
    # Bear-market summary: average 2020 + 2022 return per model
    bear_periods = {"2020 COVID", "2022 Fed"}
    print("\nBear-market summary (mean of 2020 + 2022 return):")
    models_seen = []
    for ckpt in CHECKPOINTS:
        label = ckpt["label"]
        bear_rows = [r for r in rows if r["model"] == label and r["period"] in bear_periods]
        if bear_rows:
            bear_ret  = np.mean([r["return_pct"] for r in bear_rows])
            bear_sharpe = np.mean([r["sharpe"]    for r in bear_rows])
            val_row   = next((r for r in rows if r["model"] == label and r["period"] == "2019 Val"), None)
            val_ret   = val_row["return_pct"] if val_row else float("nan")
            print(f"  {label:<30}  val={val_ret:>+6.2f}%  bear_avg={bear_ret:>+6.2f}%  bear_sharpe={bear_sharpe:>+.3f}")
            models_seen.append(label)

    # -----------------------------------------------------------------------
    # Save CSV
    if rows:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
