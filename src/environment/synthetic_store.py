"""
SyntheticEpisodeStore: loads pre-generated .npz episode files into memory
and serves them to TradingEnv on demand.

Each .npz file contains:
    stock_features  (n_stocks, lookback+ep_len, n_feat)  float32
    close_prices    (n_stocks, ep_len+1)                 float32
    market_features (lookback+ep_len, n_market_feat)     float32
    regime_probs    (3,)                                  float32
    stock_mask      (n_stocks,)                           float32

Filename convention: {method}_{nnnn}.npz
  method ∈ {negation, garch}
  Regime is inferred from regime_probs (argmax).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from src.utils.logging_config import get_logger

log = get_logger("env.synthetic_store")


class SyntheticEpisodeStore:
    """
    Loads all .npz synthetic episodes from a directory at construction time
    and groups them by dominant regime (argmax of regime_probs).

    Memory estimate: 800 episodes × (20×282×33 + 20×253 + 282×7) float32 ≈ 1.2 GB.
    If that is too large, pass lazy=True to mmap arrays on demand (slower but low RAM).
    """

    def __init__(self, episodes_dir: str, lazy: bool = False):
        self._dir   = Path(episodes_dir)
        self._lazy  = lazy
        # pool: regime_int -> list of episode dicts (or file paths if lazy)
        self._pool: dict[int, list] = {0: [], 1: [], 2: []}

        files = sorted(self._dir.glob("*.npz"))
        if not files:
            log.warning("No .npz files found in %s", episodes_dir)
            return

        for path in files:
            if lazy:
                # store path; load on sample()
                ep_meta = self._read_meta(path)
                if ep_meta is None:
                    continue
                regime = ep_meta["regime"]
                self._pool[regime].append(str(path))
            else:
                ep = self._load(path)
                if ep is None:
                    continue
                regime = int(np.argmax(ep["regime_probs"]))
                self._pool[regime].append(ep)

        counts = {r: len(v) for r, v in self._pool.items()}
        log.info(
            "SyntheticEpisodeStore loaded from %s: bull=%d  bear=%d  trans=%d",
            episodes_dir, counts[0], counts[1], counts[2],
        )

    # ------------------------------------------------------------------
    def has_regime(self, regime: int) -> bool:
        """Return True if there is at least one episode with this regime label."""
        return bool(self._pool.get(regime))

    def sample(self, regime: int, rng: np.random.Generator) -> dict:
        """
        Sample one episode for the given regime.
        Returns a dict with numpy arrays:
            stock_features, close_prices, market_features, regime_probs, stock_mask
        """
        pool = self._pool.get(regime, [])
        if not pool:
            raise ValueError(f"No synthetic episodes for regime={regime}")

        idx = int(rng.integers(0, len(pool)))
        item = pool[idx]

        if self._lazy:
            return self._load(Path(item))
        return item

    # ------------------------------------------------------------------
    @staticmethod
    def _load(path: Path) -> dict | None:
        try:
            data = np.load(str(path), allow_pickle=False)
            return {k: data[k] for k in data.files}
        except Exception as e:
            log.warning("Failed to load synthetic episode %s: %s", path.name, e)
            return None

    @staticmethod
    def _read_meta(path: Path) -> dict | None:
        """Read only regime_probs to classify without loading the full array."""
        try:
            data = np.load(str(path), allow_pickle=False)
            regime_probs = data["regime_probs"]
            return {"regime": int(np.argmax(regime_probs))}
        except Exception as e:
            log.warning("Failed to read meta from %s: %s", path.name, e)
            return None
