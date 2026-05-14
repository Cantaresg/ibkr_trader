import os
import re
from pathlib import Path
import yaml
from dotenv import load_dotenv


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML config, substituting ${VAR} placeholders from environment / .env file."""
    load_dotenv()  # loads .env into os.environ if present
    with open(config_path) as f:
        raw = f.read()
    # Replace ${VAR} with environment variable values
    def _sub(match):
        var = match.group(1)
        return os.environ.get(var, match.group(0))  # leave placeholder if not set
    raw = re.sub(r"\$\{(\w+)\}", _sub, raw)
    return yaml.safe_load(raw)


def load_universe(universe_path: str = "config/universe.yaml") -> dict[str, list[str]]:
    """Return dict mapping sector_key -> list of tickers."""
    with open(universe_path) as f:
        data = yaml.safe_load(f)
    return {k: v["tickers"] for k, v in data["sectors"].items()}


def all_tickers(universe_path: str = "config/universe.yaml") -> list[str]:
    """Flat list of all tickers in the universe."""
    universe = load_universe(universe_path)
    return [t for tickers in universe.values() for t in tickers]


def ticker_to_sector(universe_path: str = "config/universe.yaml") -> dict[str, str]:
    """Map ticker -> sector key."""
    with open(universe_path) as f:
        data = yaml.safe_load(f)
    mapping = {}
    for sector_key, sector_data in data["sectors"].items():
        for ticker in sector_data["tickers"]:
            mapping[ticker] = sector_key
    return mapping
