"""
Final v6 comparison: v4 baseline vs v6 RPPO (best/final) vs v6 PPO (best/final).
All v6 models use eod_force_flat=False.
"""
import sys
sys.path.insert(0, ".")
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-16"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import os
import pyarrow.parquet  # noqa: F401

from intraday_trader.backtester import IntradayPolicyRunner
from intraday_trader.data_store import IntradayDataStore
from src.utils.config_loader import load_config

cfg = load_config("intraday_trader/config.yaml")
ds  = IntradayDataStore(config_path="intraday_trader/config.yaml")

VARIANTS = [
    # label                           path                                                                    force_flat
    ("v4 best   (rppo) [baseline]",  "intraday_trader/checkpoints/intraday_rppo_v4/best/best_model.zip",    True),
    ("---",                            None,                                                                   True),
    ("v6 RPPO   @2M    [BEST]",       "intraday_trader/checkpoints/intraday_rppo_v6/intraday_rppo_2000000_steps.zip", False),
    ("v6 RPPO   best   [300K]",       "intraday_trader/checkpoints/intraday_rppo_v6/best/best_model.zip",   False),
    ("v6 RPPO   final  [5M]",         "intraday_trader/checkpoints/intraday_rppo_v6/final_model.zip",       False),
    ("---",                            None,                                                                   True),
    ("v6 PPO    best",                "intraday_trader/checkpoints/intraday_ppo_v6/best/best_model.zip",    False),
    ("v6 PPO    final  [5M]",         "intraday_trader/checkpoints/intraday_ppo_v6/final_model.zip",        False),
]

print(f"\n{'Model':<34} {'Sharpe':>8} {'Ann.Ret':>9} {'MDD':>7} {'WinRate':>9}")
print("=" * 72)

for label, path, force_flat in VARIANTS:
    if path is None:
        print(f"  {label}")
        continue
    if not os.path.exists(path):
        print(f"{label:<34}  (not found)")
        continue
    try:
        runner = IntradayPolicyRunner(
            model_path=path, data_store=ds,
            start_date=cfg["data"]["eval_start"],
            end_date=cfg["data"]["eval_end"],
            eod_force_flat=force_flat,
        )
        r = runner.run(n_episodes=20)["mean_metrics"]
        print(
            f"{label:<34} "
            f"{r['daily_sharpe']:>8.3f} "
            f"{r['annualised_return']:>8.1%} "
            f"{r['max_drawdown']:>7.1%} "
            f"{r['daily_win_rate']:>8.1%}"
        )
    except Exception as e:
        print(f"{label:<34}  ERROR: {e}")
