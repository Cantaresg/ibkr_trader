"""v7b sweep: N_STOCKS=10 concentration ablation."""
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

BASE = "intraday_trader/checkpoints/intraday_rppo_v7b"

SWEEP_STEPS = [
    200_000, 300_000, 400_000, 500_000, 600_000, 700_000, 800_000,
    900_000, 1_000_000, 1_250_000, 1_500_000, 1_750_000, 2_000_000,
    2_500_000, 3_000_000, 4_000_000, 5_000_000,
]

print(f"\n{'Model':<38} {'Sharpe':>8} {'Ann.Ret':>9} {'MDD':>7} {'WinRate':>9}")
print("=" * 76)
print(f"  v6 RPPO @2M [baseline]             {'2.237':>8} {'61.0%':>9} {'2.8%':>7} {'54.8%':>9}")
print(f"  v7a RPPO @200K [best]              {'1.594':>8} {'35.7%':>9} {'3.0%':>7} {'56.7%':>9}")
print("-" * 76)

for label, path in [
    ("v7b RPPO best",      f"{BASE}/best/best_model.zip"),
    ("v7b RPPO final [5M]", f"{BASE}/final_model.zip"),
]:
    if not os.path.exists(path):
        print(f"{label:<38}  (not found)")
        continue
    runner = IntradayPolicyRunner(
        model_path=path, data_store=ds,
        start_date=cfg["data"]["eval_start"],
        end_date=cfg["data"]["eval_end"],
        eod_force_flat=False,
    )
    r = runner.run(n_episodes=20)["mean_metrics"]
    print(f"{label:<38} {r['daily_sharpe']:>8.3f} {r['annualised_return']:>8.1%} {r['max_drawdown']:>7.1%} {r['daily_win_rate']:>8.1%}")

print(f"\n{'Steps':>12}  {'Sharpe':>8}  {'Ann.Ret':>8}  {'MDD':>6}  {'WinRate':>8}")
print("-" * 54)

for steps in SWEEP_STEPS:
    path = f"{BASE}/intraday_rppo_{steps}_steps.zip"
    if not os.path.exists(path):
        print(f"{steps:>12,}  (not found)")
        continue
    runner = IntradayPolicyRunner(path, ds,
        cfg["data"]["eval_start"], cfg["data"]["eval_end"], eod_force_flat=False)
    r = runner.run(n_episodes=20)["mean_metrics"]
    print(f"{steps:>12,}  {r['daily_sharpe']:>8.3f}  {r['annualised_return']:>8.1%}  {r['max_drawdown']:>6.1%}  {r['daily_win_rate']:>8.1%}")
