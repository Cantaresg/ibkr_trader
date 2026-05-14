"""
Generate synthetic bear market episodes (return-negation) for intraday training.

Each episode is saved as a .npz file in the output directory.
Episodes are used by IntradaySyntheticStore to inject bear-market diversity
into training at synthetic_ratio (default 30%) of episodes.

Usage:
    python intraday_trader/scripts/generate_synthetic.py
    python intraday_trader/scripts/generate_synthetic.py --n-negation 400 --seed 42
    python intraday_trader/scripts/generate_synthetic.py --out-dir intraday_trader/data/processed/synthetic_episodes
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pyarrow.parquet  # noqa: F401 — must precede torch/SB3 on Windows

import numpy as np

from intraday_trader.synthetic_generator import IntradaySyntheticGenerator
from src.utils.config_loader import load_config
from src.utils.logging_config import setup_logging, get_logger

log = get_logger("scripts.generate_synthetic")

_DEFAULT_OUT_DIR = "intraday_trader/data/processed/synthetic_episodes"
_DEFAULT_N       = 400


def parse_args():
    p = argparse.ArgumentParser(description="Generate intraday synthetic bear episodes")
    p.add_argument("--config",      default="intraday_trader/config.yaml")
    p.add_argument("--out-dir",     default=None, help="Output directory for .npz files")
    p.add_argument("--n-negation",  type=int, default=None, help="Number of negation episodes")
    p.add_argument("--vol-scale",   type=float, default=None, help="Bear/bull vol ratio (auto if omitted)")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--log-file",    default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)

    cfg     = load_config(args.config)
    syn_cfg = cfg.get("synthetic", {})

    out_dir   = args.out_dir or syn_cfg.get("synthetic_dir", _DEFAULT_OUT_DIR)
    n_neg     = args.n_negation or _DEFAULT_N
    vol_scale = args.vol_scale
    seed      = args.seed

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    log.info("Initialising IntradaySyntheticGenerator...")
    gen = IntradaySyntheticGenerator(config_path=args.config)

    rng     = np.random.default_rng(seed)
    n_ok    = 0
    n_fail  = 0
    batch   = 0

    log.info("Generating %d negation episodes -> %s", n_neg, out_dir)
    while n_ok < n_neg:
        batch += 1
        ep_seed = int(rng.integers(0, 2**31))
        ep = gen.generate_negation_episode(vol_scale=vol_scale, seed=ep_seed)
        if ep is None:
            n_fail += 1
            if n_fail > n_neg * 3:
                log.error("Too many generation failures (%d). Aborting.", n_fail)
                break
            continue

        fname = out_path / f"negation_{n_ok:04d}.npz"
        np.savez_compressed(
            str(fname),
            stock_features = ep["stock_features"],
            close_prices   = ep["close_prices"],
            stock_mask     = ep["stock_mask"],
        )
        n_ok += 1

        if n_ok % 50 == 0:
            log.info("  %d/%d episodes saved (failures: %d)", n_ok, n_neg, n_fail)

    log.info(
        "=== Done: %d episodes saved, %d failures. Output: %s ===",
        n_ok, n_fail, out_dir,
    )
    print(f"\nDone: {n_ok} synthetic episodes saved to {out_dir}")


if __name__ == "__main__":
    main()
