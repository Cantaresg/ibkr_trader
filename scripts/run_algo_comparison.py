"""
Run PPO vs SAC vs RecurrentPPO comparison experiments.

Each algo trains for 2M steps using the best hyperparameters found from the
PPO sweep (dd_beta=0.3, ep_len=252, regime_weights=0.35/0.45/0.20).
No synthetic augmentation — isolates the effect of the algorithm itself.

Checkpoints saved to:
    checkpoints/algo_ppo_best/
    checkpoints/algo_sac_best/
    checkpoints/algo_rppo_best/

Results appended to:
    data/processed/experiments/algo_comparison.jsonl

Usage:
    python scripts/run_algo_comparison.py             # run all pending
    python scripts/run_algo_comparison.py --list      # show table and exit
    python scripts/run_algo_comparison.py --only sac  # run one algo
    python scripts/run_algo_comparison.py --force     # re-run completed ones
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

RESULTS_PATH = Path("data/processed/experiments/algo_comparison.jsonl")
LOGS_DIR     = Path("logs/experiments")

# Best PPO hyperparameters to apply to all algos (where applicable)
_SHARED = [
    "--train-start",     "2008-01-01",
    "--train-end",       "2018-12-31",
    "--eval-start",      "2019-01-01",
    "--eval-end",        "2019-12-31",
    "--drawdown-beta",   "0.3",
    "--episode-length",  "252",
    "--regime-weights",  "0.35,0.45,0.20",
    "--timesteps",       "2000000",
]

EXPERIMENTS = [
    {
        "name":  "algo_ppo_best",
        "algo":  "ppo",
        "desc":  "PPO best-hp (dd_beta=0.3, ep252, regime-weighted) — reference",
        "extra": [],
    },
    {
        "name":  "algo_sac_best",
        "algo":  "sac",
        "desc":  "SAC best-hp (off-policy, automatic entropy, replay buffer)",
        "extra": [],
    },
    {
        "name":  "algo_rppo_best",
        "algo":  "rppo",
        "desc":  "RecurrentPPO best-hp (PPO + LSTM-256, Phase 2 architecture)",
        "extra": [],
    },
]


# ---------------------------------------------------------------------------
def load_results() -> dict[str, dict]:
    if not RESULTS_PATH.exists():
        return {}
    rows = {}
    for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            rows[row["name"]] = row
    return rows


def save_result(row: dict) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def run_experiment(exp: dict) -> dict:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{exp['name']}.log"

    cmd = [
        sys.executable, "scripts/train_agent.py",
        "--algo",     exp["algo"],
        "--run-name", exp["name"],
        "--log-file", str(log_path),
    ] + _SHARED + exp.get("extra", [])

    print(f"\n{'='*70}")
    print(f"Running: {exp['name']}  ({exp['desc']})")
    print(f"Command: {' '.join(cmd)}")
    print(f"Log: {log_path}")
    print(f"{'='*70}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    duration = int(time.time() - t0)

    best_ckpt = Path("checkpoints") / exp["name"] / "best" / "best_model.zip"
    eval_npz  = Path("checkpoints") / exp["name"] / "eval_logs" / "evaluations.npz"

    best_reward = float("nan")
    best_step   = -1
    if eval_npz.exists():
        try:
            data = np.load(str(eval_npz))
            rewards = data["results"].mean(axis=1)
            idx = int(rewards.argmax())
            best_reward = float(rewards[idx])
            best_step   = int(data["timesteps"][idx])
        except Exception as e:
            print(f"  Could not parse evaluations.npz: {e}")

    row = {
        "name":             exp["name"],
        "algo":             exp["algo"],
        "desc":             exp["desc"],
        "best_eval_reward": round(best_reward, 6),
        "best_eval_step":   best_step,
        "duration_s":       duration,
        "status":           "ok" if result.returncode == 0 else f"error:{result.returncode}",
        "timestamp":        datetime.now().isoformat(timespec="seconds"),
    }
    return row


def print_table(results: dict) -> None:
    print(f"\n{'Name':<22}  {'Algo':<6}  {'BestReward':>11}  {'BestStep':>10}  {'Duration':>9}  Status")
    print("-" * 80)
    for exp in EXPERIMENTS:
        row = results.get(exp["name"])
        if row is None:
            print(f"  {exp['name']:<20}  {exp['algo']:<6}  {'—':>11}  {'—':>10}  {'—':>9}  pending")
        else:
            dur = f"{row['duration_s']//60}m{row['duration_s']%60:02d}s"
            print(
                f"  {row['name']:<20}  {row['algo']:<6}  "
                f"{row['best_eval_reward']:>+11.4f}  "
                f"{row['best_eval_step']:>10,}  "
                f"{dur:>9}  {row['status']}"
            )
    print()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--list",  action="store_true", help="Print table and exit")
    p.add_argument("--only",  default=None,        help="Run only this algo name (ppo/sac/rppo)")
    p.add_argument("--force", action="store_true", help="Re-run even if already completed")
    return p.parse_args()


def main():
    args    = parse_args()
    results = load_results()

    if args.list:
        print_table(results)
        return

    to_run = [e for e in EXPERIMENTS
              if (args.only is None or e["algo"] == args.only)
              and (args.force or e["name"] not in results)]

    if not to_run:
        print("All experiments already completed. Use --force to re-run.")
        print_table(results)
        return

    print(f"\nAlgorithm comparison: {len(to_run)} experiment(s) to run.")
    for exp in to_run:
        row = run_experiment(exp)
        save_result(row)
        results[row["name"]] = row
        print(f"\nResult: {row}")

    print_table(results)


if __name__ == "__main__":
    main()
