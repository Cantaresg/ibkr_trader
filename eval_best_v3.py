import sys
sys.path.insert(0, ".")
import pyarrow.parquet
from intraday_trader.backtester import IntradayPolicyRunner
from intraday_trader.data_store import IntradayDataStore
from src.utils.config_loader import load_config

cfg = load_config("intraday_trader/config.yaml")
ds = IntradayDataStore(config_path="intraday_trader/config.yaml")

for label, path in [
    ("v3 best",  "intraday_trader/checkpoints/intraday_rppo_v3/best/best_model.zip"),
    ("v3 final", "intraday_trader/checkpoints/intraday_rppo_v3/final_model.zip"),
]:
    runner = IntradayPolicyRunner(
        model_path=path,
        data_store=ds,
        start_date=cfg["data"]["eval_start"],
        end_date=cfg["data"]["eval_end"],
    )
    r = runner.run(n_episodes=20)["mean_metrics"]
    print(f"\n{label}:")
    print(f"  Daily Sharpe: {r['daily_sharpe']:.3f}")
    print(f"  Ann. Return:  {r['annualised_return']:.1%}")
    print(f"  Max Drawdown: {r['max_drawdown']:.1%}")
    print(f"  Win Rate:     {r['daily_win_rate']:.1%}")
