"""
Daily data updater: refreshes OHLCV, features, scanner rankings, and regime
probabilities after market close so the next morning's inference uses
up-to-date data.

Calls the same underlying functions used by the offline pipeline scripts
(download_data.py, build_features.py, build_scanner.py, build_regime.py)
so the live and offline paths stay in sync.
"""
from __future__ import annotations
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils.config_loader import load_config, all_tickers
from src.utils.logging_config import get_logger
from src.data.ohlcv_store import download as download_ohlcv
from src.data.market_data import download_market_tickers
from src.features.pipeline import build_ticker
from src.scanner.rule_based import build_rankings
from src.regime.hmm_detector import load_model, predict_proba, save_proba
from src.features.market_features import build as build_market_features

log = get_logger("live.data_updater")


class DailyDataUpdater:
    """
    Runs once after market close (~4:30pm ET) to pull today's data and rebuild
    all derived artifacts the inference engine depends on.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.cfg         = load_config(config_path)
        self.config_path = config_path
        self.raw_dir     = self.cfg["data"]["raw_dir"]
        self.proc_dir    = self.cfg["data"]["processed_dir"]
        self.tickers     = all_tickers(self.cfg["data"]["universe_file"])

    # ------------------------------------------------------------------
    def run(self, as_of: date | None = None) -> None:
        """
        Full EOD refresh pipeline. as_of defaults to today.
        """
        today = as_of or date.today()
        log.info("Starting EOD data refresh for %s...", today)

        self._update_ohlcv(today)
        self._update_market_data(today)
        self._rebuild_features()
        self._rebuild_scanner()
        self._update_regime()

        log.info("EOD data refresh complete.")

    # ------------------------------------------------------------------
    def _update_ohlcv(self, today: date) -> None:
        log.info("Downloading OHLCV for %d tickers...", len(self.tickers))
        start = self.cfg["data"]["start_date"]
        end   = str(today + timedelta(days=1))   # yfinance end is exclusive
        for ticker in self.tickers:
            try:
                download_ohlcv(ticker, start=start, end=end,
                               raw_dir=self.raw_dir)
            except Exception as e:
                log.warning("OHLCV update failed for %s: %s", ticker, e)

    # ------------------------------------------------------------------
    def _update_market_data(self, today: date) -> None:
        log.info("Updating market data (SPY, VIX, bonds)...")
        start = self.cfg["data"]["start_date"]
        end   = str(today + timedelta(days=1))
        try:
            download_market_tickers(
                raw_dir=self.raw_dir,
                start=start,
                end=end,
            )
        except Exception as e:
            log.warning("Market data update failed: %s", e)

    # ------------------------------------------------------------------
    def _rebuild_features(self) -> None:
        log.info("Rebuilding features for %d tickers...", len(self.tickers))
        norm_window = self.cfg["features"].get("normalization_window", 252)
        ok = 0
        for ticker in self.tickers:
            try:
                result = build_ticker(
                    ticker,
                    raw_dir=self.raw_dir,
                    processed_dir=self.proc_dir,
                    norm_window=norm_window,
                    overwrite=True,
                )
                if result is not None:
                    ok += 1
            except Exception as e:
                log.warning("Feature build failed for %s: %s", ticker, e)
        log.info("Features rebuilt for %d/%d tickers.", ok, len(self.tickers))

    # ------------------------------------------------------------------
    def _rebuild_scanner(self) -> None:
        log.info("Rebuilding scanner rankings...")
        try:
            rankings = build_rankings(
                tickers=self.tickers,
                raw_dir=self.raw_dir,
                n_candidates=self.cfg["scanner"]["n_candidates"],
            )
            out_path = Path(self.proc_dir) / "scanner" / "rankings.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            rankings.to_parquet(str(out_path))
            log.info("Scanner rankings saved (%d dates).", len(rankings))
        except Exception as e:
            log.warning("Scanner rebuild failed: %s", e)

    # ------------------------------------------------------------------
    def _update_regime(self) -> None:
        log.info("Updating regime probabilities...")
        try:
            result = load_model(window_id="global")
            if result is None:
                log.warning("No fitted HMM found — skipping regime update")
                return
            model, labels = result
            market_feat = build_market_features(raw_dir=self.raw_dir)
            proba = predict_proba(model, market_feat, labels)
            save_proba(proba, window_id="global")
            log.info("Regime probabilities updated (%d dates).", len(proba))
        except Exception as e:
            log.warning("Regime update failed: %s", e)
