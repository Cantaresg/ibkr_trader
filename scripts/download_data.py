"""Phase 1: Download all OHLCV, market data, and fundamentals."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging
from src.data.downloader import download_all

if __name__ == "__main__":
    setup_logging("INFO", "logs/download.log")
    download_all()
