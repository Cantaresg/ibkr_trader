"""Phase 1: Build full feature store for all tickers."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging
from src.utils.config_loader import load_config, all_tickers
from src.features.pipeline import build_all

if __name__ == "__main__":
    setup_logging("INFO", "logs/build_features.log")
    cfg = load_config()
    tickers = all_tickers(cfg["data"]["universe_file"])
    build_all(
        tickers=tickers,
        raw_dir=cfg["data"]["raw_dir"],
        processed_dir=cfg["data"]["processed_dir"],
        norm_window=cfg["features"]["normalization_window"],
        overwrite=True,
    )
