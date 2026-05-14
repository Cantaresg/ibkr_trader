"""
update_all.py — One-shot data refresh before training or live trading.

Downloads and rebuilds all data layers in dependency order:
  1. GDELT news tone (free, no key required)
  2. Finnhub per-stock news + FinBERT sentiment (requires FINNHUB_KEY in .env)
  3. Daily OHLCV via yfinance  [EOD universe]
  4. SimFin fundamentals       [EOD universe]
  5. GDELT sentiment features rebuild
  6. Daily feature rebuild     [EOD features]
  7. Hourly OHLCV update       [intraday universe, via yfinance]
  8. Intraday feature rebuild
  9. Intraday scanner rankings rebuild

Steps that require missing API keys or optional dependencies are skipped
automatically with a warning — the rest still run.

Usage:
    python scripts/update_all.py                    # full refresh
    python scripts/update_all.py --skip-news        # skip Finnhub + GDELT download
    python scripts/update_all.py --skip-features    # data only, no feature rebuild
    python scripts/update_all.py --force-rebuild    # wipe + rebuild all feature caches
    python scripts/update_all.py --intraday-only    # only hourly + intraday features
    python scripts/update_all.py --dry-run          # print what would run, do nothing
"""
from __future__ import annotations
import sys
import os
import argparse
import subprocess
import time
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging, get_logger
from src.utils.config_loader import load_config

log = get_logger("scripts.update_all")

_PYTHON = sys.executable
_ROOT   = Path(__file__).parent.parent


def _run(
    label: str,
    cmd: list[str],
    dry_run: bool = False,
    required: bool = True,
) -> bool:
    """Run a subprocess step. Returns True on success, False on failure."""
    cmd_str = " ".join(str(c) for c in cmd)
    if dry_run:
        print(f"  [dry-run] {label}: {cmd_str}")
        return True

    log.info("=== %s ===", label)
    log.info("  %s", cmd_str)
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(_ROOT),
            capture_output=False,
            check=False,
        )
        elapsed = time.monotonic() - t0
        if result.returncode == 0:
            log.info("  %s done (%.0fs)", label, elapsed)
            return True
        else:
            msg = f"{label} exited with code {result.returncode}"
            if required:
                log.error("  %s", msg)
            else:
                log.warning("  %s (non-critical — continuing)", msg)
            return False
    except Exception as e:
        elapsed = time.monotonic() - t0
        msg = f"{label} failed after {elapsed:.0f}s: {e}"
        if required:
            log.error("  %s", msg)
        else:
            log.warning("  %s (non-critical — continuing)", msg)
        return False


def _env_key(cfg: dict, key: str) -> str:
    """Read a config key that may be an env-var placeholder like ${VAR}."""
    val = cfg["data"].get(key, "")
    if val and val.startswith("${"):
        val = os.environ.get(val[2:-1], "")
    return val or ""


def parse_args():
    p = argparse.ArgumentParser(description="Full data refresh pipeline")
    p.add_argument("--config",         default="config/config.yaml")
    p.add_argument("--intraday-config", default="intraday_trader/config.yaml")
    p.add_argument("--skip-news",      action="store_true",
                   help="Skip GDELT download + Finnhub news")
    p.add_argument("--skip-features",  action="store_true",
                   help="Skip all feature rebuild steps")
    p.add_argument("--force-rebuild",  action="store_true",
                   help="Wipe and rebuild all feature caches from scratch")
    p.add_argument("--intraday-only",  action="store_true",
                   help="Skip EOD layers; only update hourly data + intraday features")
    p.add_argument("--dry-run",        action="store_true",
                   help="Print commands without executing them")
    p.add_argument("--log-file",       default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)

    cfg = load_config(args.config)
    finnhub_key = _env_key(cfg, "finnhub_key")

    today_str  = date.today().isoformat()
    results: dict[str, bool] = {}

    def step(label: str, cmd: list[str], required: bool = False) -> bool:
        ok = _run(label, cmd, dry_run=args.dry_run, required=required)
        results[label] = ok
        return ok

    # ------------------------------------------------------------------
    # 1. GDELT news tone (free, always available)
    # ------------------------------------------------------------------
    if not args.skip_news and not args.intraday_only:
        step(
            "GDELT news update",
            [_PYTHON, "scripts/download_gdelt.py",
             "--config", args.config,
             "--end-date", today_str],
            required=False,
        )

    # ------------------------------------------------------------------
    # 2. Finnhub per-stock news + FinBERT sentiment
    # ------------------------------------------------------------------
    if not args.skip_news and not args.intraday_only:
        if finnhub_key:
            step(
                "Finnhub news + FinBERT sentiment",
                [_PYTHON, "scripts/download_news.py",
                 "--config", args.config],
                required=False,
            )
        else:
            log.warning("FINNHUB_KEY not set — skipping Finnhub news download")

    # ------------------------------------------------------------------
    # 3. Daily OHLCV (yfinance, EOD universe)
    # ------------------------------------------------------------------
    if not args.intraday_only:
        step(
            "Daily OHLCV update (yfinance)",
            [_PYTHON, "scripts/download_data.py",
             "--config", args.config],
            required=False,
        )

    # ------------------------------------------------------------------
    # 4. Fundamentals refresh
    #    SimFin is called inside download_data.py above.
    #    IBKR fundamentals are separate (requires TWS running) — skipped here;
    #    run scripts/download_ibkr_gdrive.py --skip-ohlcv manually when TWS is up.
    # ------------------------------------------------------------------
    # (no separate step — SimFin runs inside download_data.py)

    # ------------------------------------------------------------------
    # 5. GDELT feature rebuild (adds gdelt_tone / gdelt_count to daily features)
    # ------------------------------------------------------------------
    if not args.skip_features and not args.intraday_only:
        gdelt_script = Path("scripts/build_gdelt_features.py")
        if gdelt_script.exists():
            step(
                "GDELT feature rebuild",
                [_PYTHON, str(gdelt_script), "--config", args.config],
                required=False,
            )

    # ------------------------------------------------------------------
    # 6. Daily feature rebuild (OHLCV → technical indicators + fundamentals)
    # ------------------------------------------------------------------
    if not args.skip_features and not args.intraday_only:
        build_feat_cmd = [_PYTHON, "scripts/build_features.py",
                          "--config", args.config]
        if args.force_rebuild:
            build_feat_cmd.append("--force-rebuild")
        step("Daily feature rebuild", build_feat_cmd, required=False)

    # ------------------------------------------------------------------
    # 7. Hourly OHLCV update (intraday universe)
    # ------------------------------------------------------------------
    step(
        "Hourly OHLCV update (intraday)",
        [_PYTHON, "intraday_trader/scripts/update_data.py",
         "--config", args.intraday_config],
        required=False,
    )

    # ------------------------------------------------------------------
    # 8. Intraday feature rebuild
    # ------------------------------------------------------------------
    if not args.skip_features:
        intraday_feat_cmd = [_PYTHON, "intraday_trader/scripts/update_data.py",
                             "--config", args.intraday_config]
        if args.force_rebuild:
            intraday_feat_cmd.append("--force-rebuild")
        # update_data.py already rebuilds features after downloading — skip duplicate
        # step only if we did NOT already run step 7 (which rebuilds anyway)
        # So step 7 already covers this; no duplicate step needed.

    # ------------------------------------------------------------------
    # 9. Scanner rankings rebuild
    # ------------------------------------------------------------------
    if not args.skip_features:
        step(
            "Scanner rankings rebuild",
            [_PYTHON, "intraday_trader/scripts/build_scanner.py",
             "--config", args.intraday_config],
            required=False,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_ok      = sum(1 for v in results.values() if v)
    n_fail    = sum(1 for v in results.values() if not v)
    n_skipped = 0  # steps that didn't run aren't in results

    print(f"\n{'='*60}")
    print(f"  update_all.py complete: {n_ok} succeeded, {n_fail} failed")
    for label, ok in results.items():
        status = "OK  " if ok else "FAIL"
        print(f"    [{status}]  {label}")
    print(f"{'='*60}\n")

    if n_fail > 0:
        log.warning("%d step(s) failed — check logs above", n_fail)


if __name__ == "__main__":
    main()
