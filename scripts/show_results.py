"""
Print the experiment results comparison table.

Usage:
    python scripts/show_results.py
    python scripts/show_results.py --sort best    # sort by best_eval_reward
    python scripts/show_results.py --sort final   # sort by final_eval_reward
"""
import argparse
import json
from pathlib import Path

RESULTS_PATH = Path("data/processed/experiments/results.jsonl")


def load() -> list[dict]:
    if not RESULTS_PATH.exists():
        return []
    rows = {}
    for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            rows[row["name"]] = row
    return list(rows.values())


def print_table(rows: list[dict]) -> None:
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
        dur_s  = f"{dur // 60}m{dur % 60:02d}s" if dur else "--"
        cp     = r.get("changed_param", "")
        cv     = r.get("changed_value", "")
        change = "(baseline)" if cp in ("-", "", None) else f"{cp}={cv}"

        print(
            f"  {r['name']:<26}  {change:<32}  "
            f"{best_s}  {step_s}  {final_s}  {conv_s}  {dur_s:>8}  "
            f"{r.get('status', '?')}"
        )
    print("=" * W)
    print(
        "  Conv% = final/best x 100  "
        "(100% = no degradation, lower = overfitting after peak)\n"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sort", choices=["name", "best", "final"], default="name",
                   help="Sort rows by: name (default), best reward, or final reward")
    args = p.parse_args()

    rows = load()
    if not rows:
        print(f"No results found at {RESULTS_PATH}")
        print("Run:  python scripts/run_experiments.py")
        return

    key = {
        "name":  lambda r: r["name"],
        "best":  lambda r: r.get("best_eval_reward", float("-inf")),
        "final": lambda r: r.get("final_eval_reward", float("-inf")),
    }[args.sort]

    # Always keep baseline first
    baseline = [r for r in rows if r.get("status") == "reference"]
    others   = sorted([r for r in rows if r.get("status") != "reference"],
                      key=key, reverse=(args.sort != "name"))

    print_table(baseline + others)


if __name__ == "__main__":
    main()
