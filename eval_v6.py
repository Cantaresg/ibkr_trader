"""
Evaluate v6 variants on the OOS period (Nov 2025 – Apr 2026).
v6 models require eod_force_flat=False (overnight holding enabled).

Run:
  python eval_v6.py
"""
import sys
sys.path.insert(0, ".")
import os
import pyarrow.parquet  # noqa: F401

from intraday_trader.backtester import IntradayPolicyRunner
from intraday_trader.data_store import IntradayDataStore
from src.utils.config_loader import load_config

cfg = load_config("intraday_trader/config.yaml")
ds  = IntradayDataStore(config_path="intraday_trader/config.yaml")

# Baseline: best pre-v6 model (force-flat ON)
BASELINE = [
    ("v4 best  (rppo) [force-flat]", "intraday_trader/checkpoints/intraday_rppo_v4/best/best_model.zip", True),
]

# v6 models: overnight holding (force-flat OFF)
V6 = [
    ("v6 best  (rppo) [300K]",  "intraday_trader/checkpoints/intraday_rppo_v6/best/best_model.zip",               False),
    ("v6 @1M   (rppo)",         "intraday_trader/checkpoints/intraday_rppo_v6/intraday_rppo_1000000_steps.zip",   False),
    ("v6 @2M   (rppo)",         "intraday_trader/checkpoints/intraday_rppo_v6/intraday_rppo_2000000_steps.zip",   False),
    ("v6 @3M   (rppo)",         "intraday_trader/checkpoints/intraday_rppo_v6/intraday_rppo_3000000_steps.zip",   False),
    ("v6 @4M   (rppo)",         "intraday_trader/checkpoints/intraday_rppo_v6/intraday_rppo_4000000_steps.zip",   False),
    ("v6 @4.95M(rppo)",         "intraday_trader/checkpoints/intraday_rppo_v6/intraday_rppo_4950000_steps.zip",   False),
]

print(f"\n{'Model':<30} {'Sharpe':>8} {'Ann.Ret':>9} {'MDD':>8} {'WinRate':>9}")
print("-" * 68)

for label, path, force_flat in BASELINE + V6:
    if not os.path.exists(path):
        print(f"{label:<30}  (not found)")
        continue
    try:
        runner = IntradayPolicyRunner(
            model_path=path,
            data_store=ds,
            start_date=cfg["data"]["eval_start"],
            end_date=cfg["data"]["eval_end"],
            eod_force_flat=force_flat,
        )
        r = runner.run(n_episodes=20)["mean_metrics"]
        print(
            f"{label:<30} "
            f"{r['daily_sharpe']:>8.3f} "
            f"{r['annualised_return']:>8.1%} "
            f"{r['max_drawdown']:>8.1%} "
            f"{r['daily_win_rate']:>8.1%}"
        )
    except Exception as e:
        print(f"{label:<30}  ERROR: {e}")
