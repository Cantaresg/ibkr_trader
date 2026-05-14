"""
Compute SPY buy-and-hold return, max drawdown, and Sharpe for the same
three out-of-sample periods used in model evaluation.

Usage:
    python scripts/spy_benchmark.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyarrow.parquet  # noqa: F401
import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
SPY_PATH = Path(RAW_DIR) / "market" / "SPY.parquet"

PERIODS = [
    ("2019-01-01", "2019-12-31", "2019 Val"),
    ("2020-01-01", "2020-12-31", "2020 COVID"),
    ("2022-01-01", "2022-12-31", "2022 Fed"),
]

# Model results from compare_ppo_rppo.py (40 episodes, ep_len=252)
MODEL_RESULTS = [
    {"model": "ppo_nosyn",       "2019 Val": (+19.08, +1.648), "2020 COVID": (+38.28, +3.921), "2022 Fed": (+6.67, +1.102)},
    {"model": "ppo_syn50",       "2019 Val": (+17.46, +1.473), "2020 COVID": (+37.98, +3.665), "2022 Fed": (+5.73, +0.924)},
    {"model": "rppo_nosyn",      "2019 Val": (+18.60, +1.452), "2020 COVID": (+38.06, +3.875), "2022 Fed": (+6.97, +1.206)},
    {"model": "rppo_syn50(BEST)","2019 Val": (+18.35, +1.473), "2020 COVID": (+37.46, +3.986), "2022 Fed": (+7.15, +1.251)},
]


def spy_stats(spy: pd.DataFrame, start: str, end: str) -> dict:
    sub = spy.loc[start:end, "close"]
    if sub.empty:
        return {}
    total_return = (sub.iloc[-1] / sub.iloc[0]) - 1.0

    # Max drawdown from rolling peak
    roll_max = sub.cummax()
    drawdowns = (sub - roll_max) / roll_max
    max_dd = float(drawdowns.min())

    # Daily return Sharpe (annualised)
    daily_rets = sub.pct_change().dropna()
    sharpe = (daily_rets.mean() / (daily_rets.std() + 1e-10)) * np.sqrt(252)

    return {
        "return_pct": round(total_return * 100, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "sharpe":     round(sharpe, 3),
    }


def main():
    spy = pd.read_parquet(SPY_PATH)
    spy.index = pd.to_datetime(spy.index)

    print("\n" + "=" * 80)
    print("SPY buy-and-hold benchmark")
    print("=" * 80)
    spy_rows = {}
    for start, end, label in PERIODS:
        s = spy_stats(spy, start, end)
        spy_rows[label] = s
        print(
            f"  {label:<14}  return={s['return_pct']:>+7.2f}%  "
            f"maxDD={s['max_dd_pct']:>+7.2f}%  sharpe={s['sharpe']:>+.3f}"
        )

    # --- Combined comparison table ---
    print("\n" + "=" * 105)
    print("Model vs SPY — Return and Sharpe by period")
    print("=" * 105)
    header = f"  {'Model':<22}  " + "  ".join(
        f"{'':>3}{l:<14}{'Ret':>8} {'Shr':>6}" for _, _, l in PERIODS
    )
    print(f"  {'Model':<22}  {'2019 Val':>16}  {'2020 COVID':>16}  {'2022 Fed':>16}")
    print(f"  {'':22}  {'Ret%':>7} {'Sharpe':>7}  {'Ret%':>7} {'Sharpe':>7}  {'Ret%':>7} {'Sharpe':>7}")
    print("-" * 105)

    # SPY row first
    spy_vals = [spy_rows[l] for _, _, l in PERIODS]
    print(
        f"  {'SPY buy-and-hold':<22}  "
        f"{spy_vals[0]['return_pct']:>+7.2f}% {'':>7}  "
        f"{spy_vals[1]['return_pct']:>+7.2f}% {'':>7}  "
        f"{spy_vals[2]['return_pct']:>+7.2f}% {'':>7}"
    )
    print(
        f"  {'':22}  "
        f"{'':>7}  {spy_vals[0]['sharpe']:>+7.3f}  "
        f"{'':>7}  {spy_vals[1]['sharpe']:>+7.3f}  "
        f"{'':>7}  {spy_vals[2]['sharpe']:>+7.3f}"
    )
    print("-" * 105)

    for row in MODEL_RESULTS:
        vals_2019  = row["2019 Val"]
        vals_2020  = row["2020 COVID"]
        vals_2022  = row["2022 Fed"]
        def _diff(model_ret, spy_ret):
            d = model_ret - spy_ret
            return f"({d:+.1f}%)" if abs(d) >= 0.1 else ""
        print(
            f"  {row['model']:<22}  "
            f"{vals_2019[0]:>+7.2f}% {vals_2019[1]:>+7.3f}  "
            f"{vals_2020[0]:>+7.2f}% {vals_2020[1]:>+7.3f}  "
            f"{vals_2022[0]:>+7.2f}% {vals_2022[1]:>+7.3f}"
        )

    print("=" * 105)

    # --- MaxDD comparison ---
    print("\nMax Drawdown comparison (lower is better):")
    print(f"  {'Period':<14}  {'SPY MaxDD':>10}  {'Model MaxDD (all ~12%)'}")
    for start, end, label in PERIODS:
        s = spy_rows[label]
        print(f"  {label:<14}  {s['max_dd_pct']:>+9.2f}%  (models: ~12%)")

    # --- Bear-market summary ---
    print("\nBear-market alpha (model return minus SPY return):")
    for row in MODEL_RESULTS:
        covid_alpha = row["2020 COVID"][0] - spy_rows["2020 COVID"]["return_pct"]
        fed_alpha   = row["2022 Fed"][0]   - spy_rows["2022 Fed"]["return_pct"]
        avg_alpha   = (covid_alpha + fed_alpha) / 2
        print(
            f"  {row['model']:<22}  "
            f"2020={covid_alpha:>+6.1f}%  "
            f"2022={fed_alpha:>+6.1f}%  "
            f"avg_alpha={avg_alpha:>+6.1f}%"
        )


if __name__ == "__main__":
    main()
