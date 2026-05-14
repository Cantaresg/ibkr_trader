"""
Paper trading monitor — reads results/live/*.json and prints a rolling
performance table with running stats.

Usage:
    python scripts/monitor.py
    python scripts/monitor.py --results-dir results/live
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


def load_results(results_dir: Path) -> pd.DataFrame:
    records = []
    for f in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue

        date_str = data.get("date") or f.stem
        records.append({
            "date":           date_str,
            "nav":            data.get("nav"),
            "orders_total":   data.get("orders_total", data.get("orders", 0)),
            "orders_filled":  data.get("orders_filled", data.get("orders", 0)),
            "status":         data.get("status", "unknown"),
            "error":          data.get("error", ""),
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def compute_daily_returns(df: pd.DataFrame) -> pd.Series:
    nav = df["nav"].ffill()
    return nav.pct_change().fillna(0.0)


def compute_drawdown(nav_series: pd.Series) -> pd.Series:
    peak = nav_series.cummax()
    return (nav_series / peak - 1.0).clip(upper=0.0)


def _metrics_summary(returns: pd.Series, nav: pd.Series) -> dict:
    n       = len(returns)
    total   = float((1 + returns).prod() - 1)
    n_years = n / 252
    cagr    = float((1 + total) ** (1 / n_years) - 1) if n_years > 0 else float("nan")
    vol     = float(returns.std() * np.sqrt(252))
    sharpe  = float((returns.mean() * 252) / (vol + 1e-9))
    dd      = float(compute_drawdown(nav).min())
    return dict(total=total, cagr=cagr, vol=vol, sharpe=sharpe, max_dd=dd, n=n)


def print_table(df: pd.DataFrame, returns: pd.Series, drawdowns: pd.Series) -> None:
    hdr = (
        f"{'Date':<12}  {'NAV':>10}  {'Daily Ret':>10}  "
        f"{'Drawdown':>9}  {'Orders':>8}  {'Status'}"
    )
    sep = "-" * len(hdr)
    print(f"\n{sep}")
    print(hdr)
    print(sep)

    for i, row in df.iterrows():
        nav_str = f"${row['nav']:>9,.2f}" if pd.notna(row["nav"]) else f"{'N/A':>10}"
        ret_str = f"{returns.iloc[i]:>+9.2%}" if i > 0 else f"{'—':>9}"
        dd_str  = f"{drawdowns.iloc[i]:>8.2%}" if pd.notna(row["nav"]) else f"{'—':>8}"

        orders_total  = int(row["orders_total"])  if pd.notna(row["orders_total"])  else 0
        orders_filled = int(row["orders_filled"]) if pd.notna(row["orders_filled"]) else 0
        orders_str    = f"{orders_filled}/{orders_total}" if orders_total else "—"

        status = row["status"]
        flags  = ""
        if status == "error":
            flags = f"  !! {row['error'][:40]}"
        elif status == "liquidated":
            flags = "  !! LIQUIDATED"
        elif orders_total > 0 and orders_filled < orders_total:
            flags = f"  ! {orders_total - orders_filled} unfilled"

        print(
            f"  {str(row['date'].date()):<10}  {nav_str}  {ret_str}  "
            f"{dd_str}  {orders_str:>8}  {status}{flags}"
        )

    print(sep)


def print_summary(df: pd.DataFrame, returns: pd.Series, nav: pd.Series) -> None:
    if len(returns) < 2:
        print("\n  Not enough data for summary stats.\n")
        return

    m = _metrics_summary(returns.iloc[1:], nav)   # skip first day (no return)
    first = df["date"].iloc[0].date()
    last  = df["date"].iloc[-1].date()

    print(f"\n  Period: {first} to {last}  ({m['n']} trading days)")
    print(f"  Total return : {m['total']:>+.2%}")
    print(f"  CAGR         : {m['cagr']:>+.2%}")
    print(f"  Ann Volatility: {m['vol']:>7.2%}")
    print(f"  Sharpe       : {m['sharpe']:>+.2f}")
    print(f"  Max Drawdown : {m['max_dd']:>8.2%}")

    error_days = df[df["status"] == "error"]
    liq_days   = df[df["status"].isin(["liquidated", "liquidate_no_positions"])]
    if not error_days.empty:
        print(f"\n  WARNING: {len(error_days)} error day(s):")
        for _, r in error_days.iterrows():
            print(f"    {r['date'].date()} — {r['error'][:60]}")
    if not liq_days.empty:
        print(f"\n  WARNING: {len(liq_days)} liquidation day(s): "
              + ", ".join(str(r["date"].date()) for _, r in liq_days.iterrows()))
    print()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/live",
                   help="Directory containing dated JSON result files")
    return p.parse_args()


def main():
    args    = parse_args()
    rdir    = Path(args.results_dir)

    print(f"\nMonitor — reading from {rdir.resolve()}")
    print(f"As of: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    if not rdir.exists():
        print(f"  Results directory not found: {rdir}")
        return

    df = load_results(rdir)
    if df.empty:
        print("  No result files found.")
        return

    nav      = df["nav"].ffill()
    returns  = compute_daily_returns(df)
    drawdowns = compute_drawdown(nav)

    print_table(df, returns, drawdowns)
    print_summary(df, returns, nav)


if __name__ == "__main__":
    main()
