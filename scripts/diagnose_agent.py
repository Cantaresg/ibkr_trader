"""
Diagnose agent behavior: cash allocation, trade frequency, position counts.

Usage:
  python scripts/diagnose_agent.py
  python scripts/diagnose_agent.py --model intraday_trader/checkpoints/intraday_rppo_v5/best/best_model.zip
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-16"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pyarrow.parquet  # noqa: F401

import numpy as np
from intraday_trader.backtester import IntradayPolicyRunner
from intraday_trader.data_store import IntradayDataStore
from src.utils.config_loader import load_config


def diagnose(model_path: str, n_episodes: int = 20, eod_force_flat: bool = True):
    cfg = load_config("intraday_trader/config.yaml")
    ds  = IntradayDataStore(config_path="intraday_trader/config.yaml")

    runner = IntradayPolicyRunner(
        model_path=model_path,
        data_store=ds,
        start_date=cfg["data"]["eval_start"],
        end_date=cfg["data"]["eval_end"],
        eod_force_flat=eod_force_flat,
    )

    all_cash_fracs   = []
    all_stock_fracs  = []
    all_n_active     = []
    all_turnovers    = []
    all_nav_finals   = []

    print(f"\nRunning {n_episodes} diagnostic episodes on OOS period...")
    print(f"Model: {model_path}\n")

    for ep_i in range(n_episodes):
        ep = runner.run_episode(seed=ep_i)
        w  = ep["weights"]          # (n_bars, n_stocks+1)
        n_bars, n_slots = w.shape
        n_stocks = n_slots - 1

        cash_frac   = w[:, -1]                       # (n_bars,)
        stock_frac  = w[:, :n_stocks].sum(axis=1)    # (n_bars,)
        n_active    = (w[:, :n_stocks] > 0.01).sum(axis=1)  # bars with >1% in each slot

        # Turnover per bar (half the sum of absolute weight changes)
        turnover = np.abs(np.diff(w, axis=0)).sum(axis=1) / 2.0  # (n_bars-1,)

        all_cash_fracs.append(cash_frac.mean())
        all_stock_fracs.append(stock_frac.mean())
        all_n_active.append(n_active.mean())
        all_turnovers.append(turnover.mean())
        all_nav_finals.append(ep["nav"][-1])

        # Per-episode one-liner
        nav_ret = ep["nav"][-1] / ep["nav"][0] - 1
        frac_idle = (cash_frac > 0.90).mean()
        print(
            f"  Ep {ep_i+1:>2}  NAV {nav_ret:>+6.1%}  "
            f"avg_cash {cash_frac.mean():.0%}  "
            f"avg_stocks {n_active.mean():.1f}  "
            f"turnover/bar {turnover.mean():.3f}  "
            f"idle_bars {frac_idle:.0%}"
        )

    print("\n── SUMMARY ────────────────────────────────────────────")
    print(f"  Mean cash allocation:    {np.mean(all_cash_fracs):.1%}")
    print(f"  Mean stock allocation:   {np.mean(all_stock_fracs):.1%}")
    print(f"  Mean active positions:   {np.mean(all_n_active):.1f} / 20 slots")
    print(f"  Mean turnover per bar:   {np.mean(all_turnovers):.4f}  "
          f"(×7 bars = {np.mean(all_turnovers)*7:.4f}/day)")
    print(f"  Mean episode NAV return: {np.mean([n/5000-1 for n in all_nav_finals]):+.2%}")
    print(f"  Idle bar fraction (>90% cash): "
          f"{np.mean([c > 0.90 for c in all_cash_fracs]):.0%} of episodes "
          f"have mean cash > 90%")
    print("──────────────────────────────────────────────────────\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="intraday_trader/checkpoints/intraday_rppo_v4/best/best_model.zip")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--no-force-flat", action="store_true", help="Disable EOD force-flat (use for v6+ models)")
    args = p.parse_args()
    diagnose(args.model, args.episodes, eod_force_flat=not args.no_force_flat)


if __name__ == "__main__":
    main()
