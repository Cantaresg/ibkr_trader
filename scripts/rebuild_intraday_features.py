"""
Rebuild intraday features from the backfilled Google Drive OHLCV data.

Reads 1h OHLCV from the Google Drive archive (or any source_dir), recomputes
all per-ticker features, downloads fresh SPY 1h data for market features,
rebuilds market features, and saves everything to the processed directory.

Run ONCE before retraining on the extended date range.

Usage:
    python scripts/rebuild_intraday_features.py
    python scripts/rebuild_intraday_features.py --source-dir "G:/My Drive/ibkr_data/ohlcv"
    python scripts/rebuild_intraday_features.py --source-dir "G:/My Drive/ibkr_data/ohlcv" --workers 4
"""
import sys
import time
import argparse
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyarrow.parquet  # noqa: F401 — must precede torch/SB3 on Windows
import pandas as pd
import numpy as np
import yfinance as yf

from src.utils.config_loader import load_config, all_tickers
from src.utils.logging_config import setup_logging, get_logger
from intraday_trader.features import compute as compute_features, FEATURE_COLS

log = get_logger("scripts.rebuild_intraday_features")

_DEFAULT_SOURCE = "G:/My Drive/ibkr_data/ohlcv"
_CONFIG         = "intraday_trader/config.yaml"


def parse_args():
    p = argparse.ArgumentParser(description="Rebuild intraday features from backfilled OHLCV")
    p.add_argument("--source-dir",  default=_DEFAULT_SOURCE,
                   help="Directory with ticker.parquet raw 1h OHLCV files")
    p.add_argument("--config",      default=_CONFIG)
    p.add_argument("--workers",     type=int, default=1,
                   help="Parallel workers for feature computation (default 1 — safe)")
    p.add_argument("--skip-spy",    action="store_true",
                   help="Skip SPY download and market feature rebuild")
    p.add_argument("--overwrite",   action="store_true", default=True,
                   help="Overwrite existing feature files (default True)")
    p.add_argument("--log-file",    default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# SPY 1h download
# ---------------------------------------------------------------------------

def _filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    import pytz
    ET = pytz.timezone("America/New_York")
    idx_et = df.index.tz_convert(ET) if df.index.tzinfo else df.index.tz_localize("UTC").tz_convert(ET)
    return df[(idx_et.hour >= 9) & (idx_et.hour <= 15)]


def download_spy_1h(raw_ohlcv_dir: Path) -> bool:
    """Download SPY 1h (max 730d from yfinance) and save to raw_ohlcv_dir/SPY.parquet."""
    out = raw_ohlcv_dir / "SPY.parquet"
    log.info("Downloading SPY 1h (730d)...")
    try:
        df = yf.download("SPY", period="730d", interval="1h",
                         auto_adjust=True, progress=False)
        if df.empty:
            log.warning("SPY download returned empty — market features will be zeros")
            return False
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
        df = _filter_market_hours(df)
        df.dropna(how="all", inplace=True)
        df.to_parquet(out)
        log.info("SPY saved: %d bars  %s to %s", len(df), df.index[0], df.index[-1])
        return True
    except Exception as e:
        log.error("SPY download failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Per-ticker feature rebuild
# ---------------------------------------------------------------------------

def _rebuild_one(ticker: str, src_path: Path, out_path: Path) -> tuple[str, int, str]:
    """Worker: compute features for one ticker. Returns (ticker, n_bars, status)."""
    try:
        raw = pd.read_parquet(src_path)
        if raw.empty:
            return ticker, 0, "empty"
        raw.columns = [c.lower() if isinstance(c, str) else str(c).lower() for c in raw.columns]
        # Ensure market-hours only (some archived files may include pre/post market)
        import pytz
        ET = pytz.timezone("America/New_York")
        if raw.index.tzinfo is None:
            idx_et = raw.index.tz_localize("UTC").tz_convert(ET)
        else:
            idx_et = raw.index.tz_convert(ET)
        raw = raw[(idx_et.hour >= 9) & (idx_et.hour <= 15)]

        feat = compute_features(raw)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        feat.to_parquet(out_path)
        return ticker, len(feat), "ok"
    except Exception as e:
        return ticker, 0, f"error: {e}"


def rebuild_ticker_features(
    tickers: list[str],
    source_dir: Path,
    features_dir: Path,
    workers: int = 1,
    overwrite: bool = True,
) -> dict[str, int]:
    """Rebuild features for all tickers. Returns {ticker: n_bars}."""
    tasks = []
    for t in tickers:
        src = source_dir / f"{t}.parquet"
        out = features_dir / f"{t}.parquet"
        if not src.exists():
            log.warning("Source not found for %s: %s", t, src)
            continue
        if out.exists() and not overwrite:
            log.debug("Skipping %s (already exists)", t)
            continue
        tasks.append((t, src, out))

    log.info("Rebuilding features for %d tickers (workers=%d)...", len(tasks), workers)
    results = {}

    if workers <= 1:
        for i, (t, src, out) in enumerate(tasks, 1):
            ticker, n, status = _rebuild_one(t, src, out)
            results[ticker] = n
            if status == "ok":
                log.info("[%d/%d] %s: %d bars", i, len(tasks), ticker, n)
            else:
                log.warning("[%d/%d] %s: %s", i, len(tasks), ticker, status)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_rebuild_one, t, src, out): t for t, src, out in tasks}
            done = 0
            for fut in as_completed(futs):
                ticker, n, status = fut.result()
                results[ticker] = n
                done += 1
                if status == "ok":
                    log.info("[%d/%d] %s: %d bars", done, len(tasks), ticker, n)
                else:
                    log.warning("[%d/%d] %s: %s", done, len(tasks), ticker, status)

    return results


# ---------------------------------------------------------------------------
# Market features rebuild
# ---------------------------------------------------------------------------

def rebuild_market_features(raw_ohlcv_dir: Path, proc_dir: Path) -> bool:
    """Rebuild market_features_1h.parquet using SPY from raw_ohlcv_dir."""
    from intraday_trader.market_features import build as build_market
    out = proc_dir / "market_features_1h.parquet"
    raw_dir_str = str(raw_ohlcv_dir.parent)  # market_features reads raw_dir/ohlcv/SPY.parquet
    try:
        mf = build_market(raw_dir=raw_dir_str, features_dir=str(proc_dir / "features"))
        if mf is None or mf.empty:
            log.warning("Market features build returned empty")
            return False
        mf.to_parquet(out)
        log.info("Market features saved: %d bars -> %s", len(mf), out)
        return True
    except Exception as e:
        log.error("Market features rebuild failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Copy to raw_dir for scanner rebuild
# ---------------------------------------------------------------------------

def copy_to_raw_dir(source_dir: Path, raw_ohlcv_dir: Path, tickers: list[str]) -> int:
    """Copy source OHLCV files to raw_dir/ohlcv/ so build_scanner.py can read them."""
    raw_ohlcv_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for t in tickers:
        src = source_dir / f"{t}.parquet"
        dst = raw_ohlcv_dir / f"{t}.parquet"
        if src.exists():
            try:
                shutil.copy2(src, dst)
            except shutil.SameFileError:
                pass
            n += 1
    log.info("Copied %d OHLCV files to %s", n, raw_ohlcv_dir)
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)
    t0 = time.monotonic()

    cfg      = load_config(args.config)
    raw_dir  = Path(cfg.get("data", {}).get("raw_dir",       "intraday_trader/data/raw"))
    proc_dir = Path(cfg.get("data", {}).get("processed_dir", "intraday_trader/data/processed"))
    universe_file = cfg.get("universe", {}).get("file", "config/universe.yaml")

    source_dir   = Path(args.source_dir)
    features_dir = proc_dir / "features"
    raw_ohlcv_dir = raw_dir / "ohlcv"

    tickers = all_tickers(universe_file)
    log.info("Universe: %d tickers", len(tickers))

    # 1. Check source
    available = [t for t in tickers if (source_dir / f"{t}.parquet").exists()]
    missing   = [t for t in tickers if t not in available]
    log.info("Source files found: %d/%d  (missing: %s)", len(available), len(tickers), missing or "none")
    if not available:
        print(f"ERROR: No source files found in {source_dir}")
        return

    # 2. Rebuild per-ticker features
    print(f"\n[1/4] Rebuilding features from {source_dir} ...")
    results = rebuild_ticker_features(
        tickers    = available,
        source_dir = source_dir,
        features_dir = features_dir,
        workers    = args.workers,
        overwrite  = args.overwrite,
    )
    n_ok = sum(1 for v in results.values() if v > 0)
    print(f"  Done: {n_ok}/{len(available)} tickers rebuilt")
    if results:
        bar_counts = [v for v in results.values() if v > 0]
        print(f"  Bars per ticker: min={min(bar_counts)}, max={max(bar_counts)}, mean={sum(bar_counts)/len(bar_counts):.0f}")

    # 3. SPY download + market features
    if not args.skip_spy:
        print("\n[2/4] Downloading SPY 1h for market features...")
        raw_ohlcv_dir.mkdir(parents=True, exist_ok=True)
        spy_ok = download_spy_1h(raw_ohlcv_dir)

        print("\n[3/4] Rebuilding market features...")
        mf_ok = rebuild_market_features(raw_ohlcv_dir, proc_dir)
        if not mf_ok:
            print("  WARNING: Market features could not be fully rebuilt — using existing file")
    else:
        print("\n[2/4] Skipping SPY download (--skip-spy)")
        print("\n[3/4] Skipping market features rebuild")

    # 4. Copy OHLCV to raw_dir/ohlcv/ so build_scanner.py can read them
    print(f"\n[4/4] Copying OHLCV to {raw_ohlcv_dir} for scanner rebuild...")
    n_copied = copy_to_raw_dir(source_dir, raw_ohlcv_dir, available)
    print(f"  Copied {n_copied} files")

    elapsed = time.monotonic() - t0
    print(f"\n{'=' * 60}")
    print(f"  Feature rebuild complete in {elapsed/60:.1f} min")
    print(f"  Features:  {features_dir}")
    print(f"  Raw OHLCV: {raw_ohlcv_dir}  (for scanner)")
    print(f"{'=' * 60}")
    print("\nNext steps:")
    print("  1. python intraday_trader/scripts/build_scanner.py")
    print("  2. Update intraday_trader/config.yaml  start_date / train_end / eval_*")
    print("  3. python intraday_trader/scripts/train.py --algo ppo  --run-name intraday_ppo_nosyn  --synthetic-ratio 0.0  --timesteps 5000000")
    print("  4. python intraday_trader/scripts/train.py --algo ppo  --run-name intraday_ppo_syn50  --synthetic-ratio 0.3  --timesteps 5000000")
    print("  5. python intraday_trader/scripts/train.py --algo rppo --run-name intraday_rppo_nosyn --synthetic-ratio 0.0  --timesteps 5000000")
    print("  6. python intraday_trader/scripts/train.py --algo rppo --run-name intraday_rppo_syn50 --synthetic-ratio 0.3  --timesteps 5000000")


if __name__ == "__main__":
    main()
