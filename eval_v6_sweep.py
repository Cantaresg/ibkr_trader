"""Fine-grained eval sweep to find the exact peak checkpoint around 2M steps."""
import sys
sys.path.insert(0, ".")
import os
import pyarrow.parquet  # noqa: F401

from intraday_trader.backtester import IntradayPolicyRunner
from intraday_trader.data_store import IntradayDataStore
from src.utils.config_loader import load_config

cfg = load_config("intraday_trader/config.yaml")
ds  = IntradayDataStore(config_path="intraday_trader/config.yaml")

checkpoints = [500_000, 750_000, 1_000_000, 1_250_000, 1_500_000,
               1_750_000, 2_000_000, 2_250_000, 2_500_000, 2_750_000, 3_000_000]

print(f"\n{'Steps':>12}  {'Sharpe':>8}  {'Ann.Ret':>8}  {'MDD':>6}  {'WinRate':>8}")
print("-" * 54)

for steps in checkpoints:
    path = f"intraday_trader/checkpoints/intraday_rppo_v6/intraday_rppo_{steps}_steps.zip"
    if not os.path.exists(path):
        print(f"{steps:>12,}  (not found)")
        continue
    runner = IntradayPolicyRunner(
        path, ds,
        cfg["data"]["eval_start"], cfg["data"]["eval_end"],
        eod_force_flat=False,
    )
    r = runner.run(n_episodes=20)["mean_metrics"]
    print(
        f"{steps:>12,}  "
        f"{r['daily_sharpe']:>8.3f}  "
        f"{r['annualised_return']:>8.1%}  "
        f"{r['max_drawdown']:>6.1%}  "
        f"{r['daily_win_rate']:>8.1%}"
    )
