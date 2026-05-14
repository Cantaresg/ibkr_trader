"""
Quant research snapshot for a single ticker.

Usage:
  python scripts/research.py NVDA
  python scripts/research.py CRWD --date 2025-11-15
  python scripts/research.py NVDA --top10
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-16"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT         = Path(__file__).parent.parent
INTRADAY_RAW = ROOT / "intraday_trader" / "data" / "raw" / "ohlcv"
EOD_RAW      = ROOT / "data" / "raw" / "ohlcv"
SCANNER_PATH = ROOT / "intraday_trader" / "data" / "processed" / "scanner" / "intraday_rankings.parquet"
FUND_DIR     = ROOT / "data" / "raw" / "fundamentals"

_W = 70  # output width


# ── formatting helpers ────────────────────────────────────────────────────────

def _hr():
    print("═" * _W)


def _header(title: str):
    inner = f"  {title}  "
    pad = _W - len(inner)
    print("═" * (pad // 2) + inner + "═" * (pad - pad // 2))


def _pct(v: float, plus: bool = True) -> str:
    sign = "+" if (v >= 0 and plus) else ""
    return f"{sign}{v * 100:.1f}%"


def _range_bar(val: float, lo: float, hi: float, width: int = 34) -> str:
    if hi <= lo:
        return "─" * width
    pos = max(0, min(width - 1, int((val - lo) / (hi - lo) * width)))
    return "─" * pos + "▲" + "─" * (width - 1 - pos)


def _body_str(o: float, c: float, h: float, lo: float) -> str:
    rng = h - lo + 1e-8
    body_frac = abs(c - o) / rng
    n = max(1, min(5, int(body_frac * 6)))
    return ("█" * n) if c >= o else ("░" * n)


def _fmt(v, fmt: str = ".1f", suffix: str = "") -> str:
    if v is None or pd.isna(v):
        return "n/a"
    return f"{v:{fmt}}{suffix}"


# ── data loaders ──────────────────────────────────────────────────────────────

def _load_ohlcv(ticker: str) -> pd.DataFrame | None:
    """Load 1h OHLCV; fall back to EOD if not found."""
    for d in [INTRADAY_RAW, EOD_RAW]:
        for name in [ticker, ticker.replace("-", "_")]:
            p = d / f"{name}.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                df.columns = [c.lower() for c in df.columns]
                if df.index.tzinfo is None:
                    df.index = df.index.tz_localize("UTC")
                df.index = df.index.tz_convert("America/New_York")
                return df[["open", "high", "low", "close", "volume"]].sort_index()
    return None


def _slice_as_of(df: pd.DataFrame, as_of: str | None) -> pd.DataFrame:
    if as_of is None:
        return df
    cutoff = pd.Timestamp(as_of, tz="America/New_York") + pd.Timedelta(days=1, seconds=-1)
    return df[df.index <= cutoff]


def _load_scanner(as_of: str | None) -> tuple[pd.Series | None, str]:
    """Return (ranking_row, date_str) or (None, '') if unavailable."""
    if not SCANNER_PATH.exists():
        return None, ""
    ranks = pd.read_parquet(SCANNER_PATH)
    ranks.index = pd.to_datetime(ranks.index).tz_localize(None)
    cutoff = pd.Timestamp(as_of) if as_of else ranks.index[-1]
    avail = ranks.index[ranks.index <= cutoff]
    if len(avail) == 0:
        return None, ""
    row = ranks.loc[avail[-1]]
    return row, avail[-1].strftime("%Y-%m-%d")


def _load_fundamentals(ticker: str) -> pd.DataFrame | None:
    for name in [ticker, ticker.replace("-", "_")]:
        p = FUND_DIR / f"{name}.parquet"
        if p.exists():
            return pd.read_parquet(p)
    return None


# ── display sections ──────────────────────────────────────────────────────────

def _section_price(ticker: str, df: pd.DataFrame):
    last = df.iloc[-1]
    price = float(last["close"])

    today_date = df.index[-1].date()
    today_bars = df[df.index.date == today_date]
    day_open   = float(today_bars.iloc[0]["open"]) if len(today_bars) > 0 else price
    ret_today  = (price / day_open - 1) if day_open > 0 else 0.0

    daily = df.groupby(df.index.date)["close"].last()

    def _pr(n: int) -> float:
        return float(daily.iloc[-1] / daily.iloc[-1 - n] - 1) if len(daily) > n else 0.0

    _header(ticker)
    print(f"  PRICE  ${price:>10,.2f}  │  today {_pct(ret_today):>6}  │  "
          f"5d {_pct(_pr(5)):>6}  │  1m {_pct(_pr(21)):>6}  │  1y {_pct(_pr(252)):>7}")


def _section_range(df: pd.DataFrame):
    closes  = df["close"]
    recent  = closes.iloc[-252:]
    lo52, hi52 = float(recent.min()), float(recent.max())
    price   = float(closes.iloc[-1])
    prox    = (price - lo52) / (hi52 - lo52 + 1e-8)

    # ATR(14)
    h = df["high"].iloc[-14:].values
    l = df["low"].iloc[-14:].values
    c = df["close"].iloc[-14:].values
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    atr = float(tr.mean())

    bar = _range_bar(price, lo52, hi52, width=34)
    print()
    print(f"  52W  ${lo52:>9,.2f} {bar} ${hi52:>9,.2f}")
    print(f"       Proximity to 52w low: {prox * 100:.0f}%   ATR: ${atr:.2f}/bar ({atr/price*100:.1f}%)")


def _section_bars(df: pd.DataFrame):
    bars = df.tail(14).copy()
    if bars.empty:
        return

    today_date   = df.index[-1].date()
    today_bars   = df[df.index.date == today_date]
    session_high = float(today_bars["high"].max())
    session_low  = float(today_bars["low"].min())

    # Bollinger bands on last 14 closes
    c14    = df["close"].iloc[-14:].values
    bb_mid = float(c14.mean())
    bb_std = float(c14.std())
    bb_lo, bb_hi = bb_mid - 2 * bb_std, bb_mid + 2 * bb_std

    # Average volume per hour across all history
    hour_avg = df.groupby(df.index.hour)["volume"].mean()

    print()
    print(f"  {'#':>3}  {'Time':<6}  {'Open':>8}  {'High':>8}  {'Low':>8}  {'Close':>8}  {'Body':<5}  VolRel")
    for i, (ts, row) in enumerate(bars.iterrows(), 1):
        body = _body_str(row["open"], row["close"], row["high"], row["low"])
        avg_v = hour_avg.get(ts.hour, row["volume"])
        vol_r = float(row["volume"]) / (float(avg_v) + 1e-8)
        print(f"  {i:>3}  {ts.strftime('%H:%M'):<6}  "
              f"{row['open']:>8.2f}  {row['high']:>8.2f}  {row['low']:>8.2f}  {row['close']:>8.2f}  "
              f"{body:<5}  {vol_r:.1f}x")

    print(f"\n  Session floor: ${session_low:,.2f}   ceiling: ${session_high:,.2f}")
    print(f"  Bollinger:     ${bb_lo:,.2f} ── ${bb_mid:,.2f} ── ${bb_hi:,.2f}")


def _section_scanner(ticker: str, as_of: str | None):
    row, date_str = _load_scanner(as_of)
    print()
    if row is None:
        print("  SCANNER  [rankings file not found]")
        return

    tickers = row.dropna().tolist()
    if ticker in tickers:
        rank = tickers.index(ticker) + 1
        status = f"Rank {rank} / 20   ✓  SELECTED"
    else:
        status = "NOT in today's top-20"

    print(f"  SCANNER RANKING — {date_str}")
    print(f"  {status}")
    print(f"  Top-5 today: {', '.join(str(t) for t in tickers[:5])}")


def _section_fundamentals(ticker: str):
    df = _load_fundamentals(ticker)
    print()
    if df is None:
        print("  FUNDAMENTALS  [not available]")
        return

    row = df.iloc[-1]
    date_str = df.index[-1].strftime("%Y-%m-%d") if hasattr(df.index[-1], "strftime") else str(df.index[-1])
    pe  = _fmt(row.get("pe_ratio"),       ".1f", "x")
    pb  = _fmt(row.get("pb_ratio"),       ".1f", "x")
    de  = _fmt(row.get("debt_to_equity"), ".1f")
    roe = _fmt(row.get("roe"),            ".1f", "%")
    rg_raw = row.get("revenue_growth")
    rg  = f"{rg_raw * 100:+.1f}%" if rg_raw is not None and not np.isnan(float(rg_raw)) else "n/a"

    print(f"  FUNDAMENTALS (as of {date_str})")
    print(f"  P/E: {pe:<10}  P/B: {pb:<10}  Debt/Equity: {de}")
    print(f"  ROE: {roe:<10}  Revenue growth YoY: {rg}")


# ── top-10 command ────────────────────────────────────────────────────────────

def _cmd_top10(as_of: str | None):
    row, date_str = _load_scanner(as_of)
    if row is None:
        print("Scanner rankings file not found.")
        return
    tickers = row.dropna().tolist()
    _header(f"TOP-10 SCANNER PICKS — {date_str}")
    for i, t in enumerate(tickers[:10], 1):
        print(f"  {i:>2}.  {t}")
    _hr()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Per-ticker quant research snapshot")
    parser.add_argument("ticker", nargs="?", help="Ticker symbol (e.g. NVDA, CRWD, GRRR)")
    parser.add_argument("--date",   default=None, metavar="YYYY-MM-DD",
                        help="As-of date; defaults to latest available")
    parser.add_argument("--top10",  action="store_true",
                        help="Show top-10 scanner picks for the date instead of a single ticker")
    args = parser.parse_args()

    if args.top10:
        _cmd_top10(args.date)
        return

    if not args.ticker:
        parser.print_help()
        return

    ticker = args.ticker.upper()

    df = _load_ohlcv(ticker)
    if df is None:
        print(f"'{ticker}': no OHLCV data found in {INTRADAY_RAW} or {EOD_RAW}")
        sys.exit(1)

    df = _slice_as_of(df, args.date)
    if df.empty:
        print(f"'{ticker}': no data on or before {args.date}")
        sys.exit(1)

    _section_price(ticker, df)
    _section_range(df)
    _section_bars(df)
    _section_scanner(ticker, args.date)
    _section_fundamentals(ticker)
    _hr()


if __name__ == "__main__":
    main()
