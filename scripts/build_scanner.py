"""Build market features, stock scanner rankings, and HMM regime states."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging, get_logger
from src.utils.config_loader import load_config, all_tickers
from src.features.market_features import build as build_market_features
from src.scanner.rule_based import build_rankings
from src.scanner.scanner_store import save as save_scanner
from src.regime.hmm_detector import fit, predict_proba, save_model, save_proba

log = get_logger("build_scanner")

if __name__ == "__main__":
    setup_logging("INFO", "logs/build_scanner.log")
    cfg = load_config()
    raw_dir = cfg["data"]["raw_dir"]
    tickers = all_tickers(cfg["data"]["universe_file"])

    log.info("=== Building market features ===")
    mkt = build_market_features(tickers, raw_dir, cache=True, overwrite=False)
    log.info("Market features: %d rows x %d cols", len(mkt), len(mkt.columns))

    log.info("=== Building scanner rankings ===")
    rankings = build_rankings(tickers, raw_dir, n_candidates=cfg["scanner"]["n_candidates"])
    save_scanner(rankings)
    log.info("Scanner rankings saved: %d dates", len(rankings))

    log.info("=== Fitting HMM regime detector (global) ===")
    model, labels = fit(mkt, n_states=cfg["regime"]["n_states"],
                        n_iter=cfg["regime"]["n_iter"], seed=cfg["project"]["seed"])
    save_model(model, labels, window_id="global")

    log.info("=== Computing regime probabilities ===")
    proba = predict_proba(model, mkt, labels)
    save_proba(proba, window_id="global")

    # Quick sanity check: show regime distribution
    dominant = proba.idxmax(axis=1).value_counts(normalize=True)
    log.info("Regime distribution: %s", dominant.to_dict())
    log.info("=== Done ===")
