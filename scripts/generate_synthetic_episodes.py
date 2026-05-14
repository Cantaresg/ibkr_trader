"""
Offline batch generator for synthetic bear episodes.

Usage:
    python scripts/generate_synthetic_episodes.py
    python scripts/generate_synthetic_episodes.py --n-negation 400 --n-garch 400
    python scripts/generate_synthetic_episodes.py --n-negation 200 --n-garch 0  # negation only
    python scripts/generate_synthetic_episodes.py --out-dir data/processed/synthetic_episodes

Output:
    data/processed/synthetic_episodes/
        negation_{i:04d}.npz    — Method A episodes
        garch_{i:04d}.npz       — Method B episodes

Each .npz contains arrays:
    stock_features  (n_stocks, lookback+ep_len, 33)   float32
    close_prices    (n_stocks, ep_len+1)               float32
    market_features (lookback+ep_len, 10)              float32
    regime_probs    (3,)                               float32
    stock_mask      (n_stocks,)                        float32
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyarrow.parquet  # noqa: F401  — must precede torch/SB3

import numpy as np

from src.data.synthetic_generator import SyntheticGenerator
from src.utils.logging_config import setup_logging, get_logger

log = get_logger("generate_synthetic")


def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic bear episodes for training augmentation")
    p.add_argument("--config",       default="config/config.yaml")
    p.add_argument("--n-negation",   type=int, default=400, help="Number of return-negation episodes")
    p.add_argument("--n-garch",      type=int, default=400, help="Number of GARCH Monte Carlo episodes")
    p.add_argument("--vol-scale",    type=float, default=None,
                   help="Bear/bull vol ratio for negation (default: auto from HMM-labelled data)")
    p.add_argument("--out-dir",      default="data/processed/synthetic_episodes")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--log-file",     default=None)
    return p.parse_args()


def save_episode(ep: dict, path: Path) -> None:
    np.savez_compressed(
        str(path),
        stock_features  = ep["stock_features"],
        close_prices    = ep["close_prices"],
        market_features = ep["market_features"],
        regime_probs    = ep["regime_probs"],
        stock_mask      = ep["stock_mask"],
    )


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    log.info("Initialising SyntheticGenerator (loads all OHLCV + features)...")
    gen = SyntheticGenerator(config_path=args.config)

    rng = np.random.default_rng(args.seed)

    # ---------------------------------------------------------------
    # Method A — Return Negation
    # ---------------------------------------------------------------
    if args.n_negation > 0:
        log.info("=== Generating %d negation episodes ===", args.n_negation)
        n_ok = 0
        n_fail = 0
        for i in range(args.n_negation):
            ep = gen.generate_negation_episode(
                vol_scale=args.vol_scale,
                seed=int(rng.integers(0, 2**31)),
            )
            if ep is None:
                n_fail += 1
                if n_fail > args.n_negation // 2:
                    log.warning("Too many failures in negation generation — stopping early")
                    break
                continue
            path = out_dir / f"negation_{n_ok:04d}.npz"
            save_episode(ep, path)
            n_ok += 1
            if n_ok % 50 == 0:
                log.info("  Negation: %d/%d saved", n_ok, args.n_negation)
        log.info("Negation episodes: %d saved, %d failed", n_ok, n_fail)

    # ---------------------------------------------------------------
    # Method B — GARCH Monte Carlo
    # ---------------------------------------------------------------
    if args.n_garch > 0:
        log.info("=== Generating %d GARCH episodes ===", args.n_garch)
        n_ok = 0
        n_fail = 0
        for i in range(args.n_garch):
            ep = gen.generate_garch_episode(seed=int(rng.integers(0, 2**31)))
            if ep is None:
                n_fail += 1
                if n_fail > args.n_garch // 2:
                    log.warning("Too many failures in GARCH generation — stopping early")
                    break
                continue
            path = out_dir / f"garch_{n_ok:04d}.npz"
            save_episode(ep, path)
            n_ok += 1
            if n_ok % 50 == 0:
                log.info("  GARCH: %d/%d saved", n_ok, args.n_garch)
        log.info("GARCH episodes: %d saved, %d failed", n_ok, n_fail)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    all_files = sorted(out_dir.glob("*.npz"))
    negation_files = [f for f in all_files if f.stem.startswith("negation")]
    garch_files    = [f for f in all_files if f.stem.startswith("garch")]
    log.info(
        "=== Generation complete ===\n"
        "  Negation episodes : %d\n"
        "  GARCH episodes    : %d\n"
        "  Total             : %d\n"
        "  Output directory  : %s",
        len(negation_files), len(garch_files), len(all_files), out_dir,
    )


if __name__ == "__main__":
    main()
