"""
Evaluate v5 variants on the OOS period (Nov 2025 – Apr 2026).
Run after training completes:
  python eval_v5.py
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

VARIANTS = [
    ("v4 best  (rppo)",       "intraday_trader/checkpoints/intraday_rppo_v4/best/best_model.zip"),
    ("v4 final (rppo)",       "intraday_trader/checkpoints/intraday_rppo_v4/final_model.zip"),
    ("v5 best  (rppo)",       "intraday_trader/checkpoints/intraday_rppo_v5/best/best_model.zip"),
    ("v5 @4.9M (rppo)",       "intraday_trader/checkpoints/intraday_rppo_v5/intraday_rppo_4900000_steps.zip"),
]

print(f"\n{'Model':<26} {'Sharpe':>8} {'Ann.Ret':>9} {'MDD':>8} {'WinRate':>9}")
print("-" * 62)

for label, path in VARIANTS:
    if not os.path.exists(path):
        print(f"{label:<26}  (not found)")
        continue
    try:
        runner = IntradayPolicyRunner(
            model_path=path,
            data_store=ds,
            start_date=cfg["data"]["eval_start"],
            end_date=cfg["data"]["eval_end"],
        )
        r = runner.run(n_episodes=20)["mean_metrics"]
        print(
            f"{label:<26} "
            f"{r['daily_sharpe']:>8.3f} "
            f"{r['annualised_return']:>8.1%} "
            f"{r['max_drawdown']:>8.1%} "
            f"{r['daily_win_rate']:>8.1%}"
        )
    except Exception as e:
        print(f"{label:<26}  ERROR: {e}")
