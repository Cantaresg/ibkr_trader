"""
Build and save intraday daily stock scanner rankings.

Reads 1h OHLCV parquet files (from download_initial_1h.py) and computes
prior-day composite scores for all universe stocks, outputting a parquet
rankings file used by IntradayDataStore at training / backtest time.

Usage:
    python intraday_trader/scripts/build_scanner.py
    python intraday_trader/scripts/build_scanner.py --n-candidates 20
    python intraday_trader/scripts/build_scanner.py --config intraday_trader/config.yaml
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pyarrow.parquet  # noqa: F401 — must precede torch/SB3 on Windows

from intraday_trader.constants import INTRADAY_UNIVERSE, UNIVERSE_FILE
from intraday_trader.scanner import (
    build_intraday_rankings,
    build_rankings,
    save_rankings,
    _INTRADAY_RANKINGS_PATH,
    _RANKINGS_PATH,
)
from src.utils.config_loader import load_config, all_tickers
from src.utils.logging_config import setup_logging, get_logger

log = get_logger("scripts.build_scanner")


def parse_args():
    p = argparse.ArgumentParser(description="Build intraday scanner rankings")
    p.add_argument("--config",       default="intraday_trader/config.yaml")
    p.add_argument("--out",          default=None, help="Output path (default: from config)")
    p.add_argument("--intraday-out", default=None, help="Output path for intraday scanner rankings")
    p.add_argument("--n-candidates", type=int, default=None)
    p.add_argument("--mode",         choices=["daily", "intraday", "both"], default="both")
    p.add_argument("--log-file",     default=None)
    return p.parse_args()


def _load_universe(cfg: dict) -> list[str]:
    universe_file = cfg.get("universe", {}).get("file", UNIVERSE_FILE)
    try:
        tickers = all_tickers(universe_file)
        log.info("Loaded %d tickers from %s", len(tickers), universe_file)
        return tickers
    except Exception as e:
        log.warning("Could not load %s (%s) — using fallback list", universe_file, e)
        return list(INTRADAY_UNIVERSE)


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)

    cfg           = load_config(args.config)
    raw_dir       = cfg.get("data", {}).get("raw_dir", "intraday_trader/data/raw")
    scanner_cfg   = cfg.get("scanner", {})
    intraday_cfg  = scanner_cfg.get("intraday", {})
    n_candidates  = args.n_candidates or scanner_cfg.get("n_candidates", 20)
    out_path      = args.out or scanner_cfg.get("rankings_path", _RANKINGS_PATH)
    intraday_out_path = args.intraday_out or intraday_cfg.get("rankings_path", _INTRADAY_RANKINGS_PATH)

    # Signal weights
    weights = scanner_cfg.get("weights", {})
    w_mom   = weights.get("momentum",         0.35)
    w_vol   = weights.get("volume",           0.35)
    w_rec   = weights.get("recovery",         0.30)
    w_prox  = weights.get("proximity_to_low", 0.0)
    rec_thr = scanner_cfg.get("recovery_threshold", 0.03)

    tickers = _load_universe(cfg)

    log.info(
        "Building rankings: n_candidates=%d  weights=(mom=%.2f, vol=%.2f, rec=%.2f, prox=%.2f)  recovery_thr=%.3f",
        n_candidates, w_mom, w_vol, w_rec, w_prox, rec_thr,
    )

    if args.mode in ("daily", "both"):
        rankings = build_rankings(
            tickers            = tickers,
            raw_dir            = raw_dir,
            n_candidates       = n_candidates,
            w_momentum         = w_mom,
            w_volume           = w_vol,
            w_recovery         = w_rec,
            w_proximity        = w_prox,
            recovery_threshold = rec_thr,
        )
        save_rankings(rankings, path=out_path)
        log.info("Daily scanner saved to: %s", out_path)
        print(f"Daily scanner: {len(rankings)} rows × {n_candidates} candidates -> {out_path}")

    if args.mode in ("intraday", "both"):
        intraday_rankings = build_intraday_rankings(
            tickers=tickers,
            raw_dir=raw_dir,
            n_candidates=n_candidates,
            lookback_hours=intraday_cfg.get("lookback_hours", 3),
            w_momentum=intraday_cfg.get("weights", {}).get("momentum", 0.60),
            w_volume=intraday_cfg.get("weights", {}).get("volume", 0.25),
            w_stability=intraday_cfg.get("weights", {}).get("stability", 0.15),
            min_history_bars=intraday_cfg.get("min_history_bars", scanner_cfg.get("min_history_bars", 200)),
        )
        save_rankings(intraday_rankings, path=intraday_out_path)
        log.info("Intraday scanner saved to: %s", intraday_out_path)
        print(f"Intraday scanner: {len(intraday_rankings)} rows × {n_candidates} candidates -> {intraday_out_path}")


if __name__ == "__main__":
    main()
