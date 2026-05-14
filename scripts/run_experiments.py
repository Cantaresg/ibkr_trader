"""
Controlled hyperparameter experiments -- one change at a time vs. Phase 1 baseline.

Each experiment changes exactly ONE parameter; everything else is kept at Phase 1
defaults (2M steps, seed=42, train 2013-2018, eval 2019).

Results are appended to data/processed/experiments/results.jsonl so runs
are resumable. Use show_results.py to view the comparison table at any time.

Usage:
    python scripts/run_experiments.py            # run all pending experiments
    python scripts/run_experiments.py --list     # print table and exit
    python scripts/run_experiments.py --only exp_01_cap_1m
    python scripts/run_experiments.py --force    # re-run even if completed
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Reference row -- Phase 1 results already completed, no need to re-run
# ---------------------------------------------------------------------------
BASELINE = {
    "name":              "baseline_phase1",
    "desc":              "Phase 1 baseline -- 2M steps, kl=0.02, dd_beta=2.0",
    "changed_param":     "-",
    "changed_value":     "-",
    "best_eval_reward":  -0.462286,
    "best_eval_step":    700_000,
    "best_eval_std":     0.705907,
    "final_eval_reward": -1.399768,
    "final_eval_std":    1.074341,
    "n_evals":           40,
    "duration_s":        2271,
    "status":            "reference",
    "timestamp":         "2026-04-28T00:00:00",
}

# ---------------------------------------------------------------------------
# Experiment definitions -- add new rows here to extend the suite
# ---------------------------------------------------------------------------
# All experiments pin 2M steps to match the baseline, except exp_01 (tests the cap).
_S = ["--timesteps", "2000000"]

EXPERIMENTS = [
    # --- Training budget ---
    {
        "name": "exp_01_cap_1m", "desc": "Cap at 1M steps (baseline=2M)",
        "changed_param": "total_timesteps", "changed_value": 1_000_000,
        "extra_args": ["--timesteps", "1000000"],
    },
    # --- KL threshold ---
    {
        "name": "exp_02_kl_005", "desc": "target_kl=0.05 (baseline=0.02)",
        "changed_param": "target_kl", "changed_value": 0.05,
        "extra_args": ["--target-kl", "0.05"] + _S,
    },
    {
        "name": "exp_03_kl_010", "desc": "target_kl=0.10 (baseline=0.02)",
        "changed_param": "target_kl", "changed_value": 0.10,
        "extra_args": ["--target-kl", "0.10"] + _S,
    },
    # --- Drawdown penalty weight ---
    {
        "name": "exp_04_dd_beta_10", "desc": "drawdown_beta=1.0 (baseline=2.0)",
        "changed_param": "drawdown_penalty_weight", "changed_value": 1.0,
        "extra_args": ["--drawdown-beta", "1.0"] + _S,
    },
    {
        "name": "exp_05_dd_beta_05", "desc": "drawdown_beta=0.5 (baseline=2.0)",
        "changed_param": "drawdown_penalty_weight", "changed_value": 0.5,
        "extra_args": ["--drawdown-beta", "0.5"] + _S,
    },
    # --- Drawdown penalty threshold ---
    {
        "name": "exp_06_dd_thresh_010", "desc": "drawdown_threshold=0.10 (baseline=0.05)",
        "changed_param": "drawdown_threshold", "changed_value": 0.10,
        "extra_args": ["--drawdown-threshold", "0.10"] + _S,
    },
    {
        "name": "exp_07_dd_thresh_015", "desc": "drawdown_threshold=0.15 (baseline=0.05)",
        "changed_param": "drawdown_threshold", "changed_value": 0.15,
        "extra_args": ["--drawdown-threshold", "0.15"] + _S,
    },
    # --- Entropy coefficient ---
    {
        "name": "exp_08_ent_001", "desc": "ent_coef=0.01 (baseline=0.03)",
        "changed_param": "ent_coef", "changed_value": 0.01,
        "extra_args": ["--ent-coef", "0.01"] + _S,
    },
    {
        "name": "exp_09_ent_010", "desc": "ent_coef=0.10 (baseline=0.03)",
        "changed_param": "ent_coef", "changed_value": 0.10,
        "extra_args": ["--ent-coef", "0.10"] + _S,
    },
    # --- Rollout length ---
    {
        "name": "exp_10_nsteps_4096", "desc": "n_steps=4096 (baseline=2048)",
        "changed_param": "n_steps", "changed_value": 4096,
        "extra_args": ["--n-steps", "4096"] + _S,
    },
    # --- Discount factor ---
    {
        "name": "exp_11_gamma_095", "desc": "gamma=0.95 (baseline=0.99)",
        "changed_param": "gamma", "changed_value": 0.95,
        "extra_args": ["--gamma", "0.95"] + _S,
    },
    # --- Warm-start from peak checkpoint instead of final model ---
    {
        "name": "exp_12_warmstart_700k", "desc": "Warm-start from Phase1 700K checkpoint (baseline=none)",
        "changed_param": "warm_start", "changed_value": "ppo_700000_steps",
        "extra_args": ["--warm-start", "checkpoints/phase1_baseline/ppo_700000_steps.zip"] + _S,
    },
    # --- Learning rate ---
    {
        "name": "exp_13_lr_3e5", "desc": "lr=3e-5 (baseline=1e-4)",
        "changed_param": "learning_rate", "changed_value": 3e-5,
        "extra_args": ["--lr", "3e-5"] + _S,
    },
    {
        "name": "exp_14_lr_3e4", "desc": "lr=3e-4 (baseline=1e-4, SB3 default)",
        "changed_param": "learning_rate", "changed_value": 3e-4,
        "extra_args": ["--lr", "3e-4"] + _S,
    },
    # --- Episode length ---
    {
        "name": "exp_15_ep_len_126", "desc": "episode_length=126 half-year (baseline=252)",
        "changed_param": "episode_length", "changed_value": 126,
        "extra_args": ["--episode-length", "126"] + _S,
    },
    # --- Model size ---
    {
        "name": "exp_16_features_256", "desc": "features_dim=256 (baseline=512)",
        "changed_param": "features_dim", "changed_value": 256,
        "extra_args": ["--features-dim", "256"] + _S,
    },
    # --- Clip range ---
    {
        "name": "exp_17_clip_01", "desc": "clip_range=0.1 (baseline=0.2)",
        "changed_param": "clip_range", "changed_value": 0.1,
        "extra_args": ["--clip-range", "0.1"] + _S,
    },
    {
        "name": "exp_18_clip_03", "desc": "clip_range=0.3 (baseline=0.2)",
        "changed_param": "clip_range", "changed_value": 0.3,
        "extra_args": ["--clip-range", "0.3"] + _S,
    },
    # --- Batch size ---
    {
        "name": "exp_19_batch_512", "desc": "batch_size=512 (baseline=256)",
        "changed_param": "batch_size", "changed_value": 512,
        "extra_args": ["--batch-size", "512"] + _S,
    },
    # --- Value function coefficient ---
    {
        "name": "exp_20_vf_coef_10", "desc": "vf_coef=1.0 (baseline=0.5)",
        "changed_param": "vf_coef", "changed_value": 1.0,
        "extra_args": ["--vf-coef", "1.0"] + _S,
    },
    # --- Drawdown penalty fine-tuning (follow-up to exp_04/05) ---
    {
        "name": "exp_21_dd_beta_04", "desc": "drawdown_beta=0.4 (baseline=2.0)",
        "changed_param": "drawdown_penalty_weight", "changed_value": 0.4,
        "extra_args": ["--drawdown-beta", "0.4"] + _S,
    },
    {
        "name": "exp_22_dd_beta_03", "desc": "drawdown_beta=0.3 (baseline=2.0)",
        "changed_param": "drawdown_penalty_weight", "changed_value": 0.3,
        "extra_args": ["--drawdown-beta", "0.3"] + _S,
    },
    # --- Synthetic bear augmentation (best hp: dd_beta=0.3, ep_len=126, bear_weight=0.45) ---
    # Ablation over synthetic_ratio: how much synthetic bear data is optimal?
    # All 3 use the same mixed pool (negation+GARCH); differ only in ratio.
    {
        "name": "syn_01_ratio_25",
        "desc": "25% bear episodes synthetic (light augmentation)",
        "changed_param": "synthetic_ratio",
        "changed_value": 0.25,
        "extra_args": [
            "--regime-weights", "0.35,0.45,0.20",
            "--drawdown-beta", "0.3",
            "--episode-length", "126",
            "--synthetic-dir", "data/processed/synthetic_episodes",
            "--synthetic-ratio", "0.25",
        ] + _S,
    },
    {
        "name": "syn_02_ratio_50",
        "desc": "50% bear episodes synthetic (moderate augmentation)",
        "changed_param": "synthetic_ratio",
        "changed_value": 0.50,
        "extra_args": [
            "--regime-weights", "0.35,0.45,0.20",
            "--drawdown-beta", "0.3",
            "--episode-length", "126",
            "--synthetic-dir", "data/processed/synthetic_episodes",
            "--synthetic-ratio", "0.50",
        ] + _S,
    },
    {
        "name": "syn_03_ratio_75",
        "desc": "75% bear episodes synthetic (heavy augmentation)",
        "changed_param": "synthetic_ratio",
        "changed_value": 0.75,
        "extra_args": [
            "--regime-weights", "0.35,0.45,0.20",
            "--drawdown-beta", "0.3",
            "--episode-length", "126",
            "--synthetic-dir", "data/processed/synthetic_episodes",
            "--synthetic-ratio", "0.75",
        ] + _S,
    },
]

RESULTS_PATH = Path("data/processed/experiments/results.jsonl")
LOGS_DIR     = Path("logs/experiments")


# ---------------------------------------------------------------------------
# Helpers
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


def extract_npz_metrics(run_name: str) -> dict:
    npz = Path(f"checkpoints/{run_name}/eval_logs/evaluations.npz")
    if not npz.exists():
        return {}
    d       = np.load(npz)
    rewards = d["results"].mean(axis=1)
    stds    = d["results"].std(axis=1)
    steps   = d["timesteps"]
    best_i  = int(rewards.argmax())
    return {
        "best_eval_reward":  float(rewards[best_i]),
        "best_eval_step":    int(steps[best_i]),
        "best_eval_std":     float(stds[best_i]),
        "final_eval_reward": float(rewards[-1]),
        "final_eval_std":    float(stds[-1]),
        "n_evals":           int(len(rewards)),
    }


def print_table(existing: dict[str, dict]) -> None:
    rows = [BASELINE] + [
        existing.get(e["name"], {
            "name":   e["name"],
            "desc":   e["desc"],
            "changed_param": e["changed_param"],
            "changed_value": e["changed_value"],
            "status": "pending",
        })
        for e in EXPERIMENTS
    ]

    W = 108
    print(f"\n{'Experiment Results':^{W}}")
    print("=" * W)
    print(
        f"  {'Name':<26}  {'Changed':<32}  "
        f"{'Best Reward':>11}  {'@ Step':>9}  "
        f"{'Final Reward':>12}  {'Conv%':>6}  {'Time':>8}  Status"
    )
    print("-" * W)
    for r in rows:
        best  = r.get("best_eval_reward")
        final = r.get("final_eval_reward")
        step  = r.get("best_eval_step")
        dur   = r.get("duration_s", 0)

        best_s  = f"{best:>11.4f}"  if isinstance(best,  float) else f"{'N/A':>11}"
        final_s = f"{final:>12.4f}" if isinstance(final, float) else f"{'N/A':>12}"
        step_s  = f"{step:>9,}"     if isinstance(step,  int)   else f"{'N/A':>9}"
        conv_s  = (
            f"{(final / best * 100):>6.1f}"
            if isinstance(best, float) and isinstance(final, float) and best != 0
            else f"{'N/A':>6}"
        )
        dur_s   = f"{dur // 60}m{dur % 60:02d}s" if dur else "--"
        cp = r.get('changed_param', '--')
        cv = r.get('changed_value', '--')
        change  = "(baseline)" if cp == "-" else f"{cp}={cv}"

        print(
            f"  {r['name']:<26}  {change:<32}  "
            f"{best_s}  {step_s}  {final_s}  {conv_s}  {dur_s:>8}  "
            f"{r.get('status', 'pending')}"
        )
    print("=" * W)
    print(
        "  Conv% = final/best x 100  "
        "(100% = no degradation after peak, lower = overfitting)\n"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_experiment(exp: dict) -> dict:
    name     = exp["name"]
    log_path = LOGS_DIR / f"{name}.log"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "scripts/train_agent.py",
        "--run-name",    name,
        "--train-start", "2013-01-01",
        "--train-end",   "2018-12-31",
        "--eval-start",  "2019-01-01",
        "--eval-end",    "2019-12-31",
        "--log-file",    str(log_path),
    ] + exp["extra_args"]

    print(f"\n{'=' * 64}")
    print(f"  Experiment : {name}")
    print(f"  Change     : {exp['changed_param']} = {exp['changed_value']}")
    print(f"  Log        : {log_path}")
    print(f"  Command    : {' '.join(cmd[2:])}")
    print(f"{'=' * 64}", flush=True)

    t0   = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )
    stdout_enc = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
    for line in proc.stdout:
        sys.stdout.buffer.write(line.encode(stdout_enc, errors="replace"))
        sys.stdout.buffer.flush()
    proc.wait()
    duration = int(time.time() - t0)

    metrics = extract_npz_metrics(name)
    row = {
        "name":          name,
        "desc":          exp["desc"],
        "changed_param": exp["changed_param"],
        "changed_value": exp["changed_value"],
        "status":        "ok" if proc.returncode == 0 else f"failed(rc={proc.returncode})",
        "duration_s":    duration,
        "timestamp":     datetime.now().isoformat(),
        **metrics,
    }
    save_result(row)
    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Run controlled hyperparameter experiments")
    p.add_argument("--list",  action="store_true", help="Show results table and exit")
    p.add_argument("--only",  default=None,
                   help="Run only named experiment(s), comma-separated: 'syn_01,syn_02'")
    p.add_argument("--force", action="store_true", help="Re-run even if already completed")
    args = p.parse_args()

    existing = load_results()

    # Seed baseline reference if not already present
    if BASELINE["name"] not in existing:
        save_result(BASELINE)
        existing[BASELINE["name"]] = BASELINE

    if args.list:
        print_table(existing)
        return

    to_run = EXPERIMENTS
    if args.only:
        names = {n.strip() for n in args.only.split(",")}
        to_run = [e for e in EXPERIMENTS if e["name"] in names]
        if not to_run:
            print(f"Unknown experiment(s) '{args.only}'. "
                  f"Valid names: {[e['name'] for e in EXPERIMENTS]}")
            sys.exit(1)

    for exp in to_run:
        if not args.force and exp["name"] in existing:
            print(f"  Skipping {exp['name']} (already in results -- use --force to re-run)")
            continue
        result = run_experiment(exp)
        existing[result["name"]] = result
        print_table(existing)


if __name__ == "__main__":
    main()
