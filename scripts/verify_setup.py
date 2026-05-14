"""Phase 0 verification — run this to confirm the environment is ready."""
import sys
import importlib

def check(label, fn):
    try:
        result = fn()
        print(f"  [OK] {label}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False

print("=== IBKR DRL Trader — Environment Verification ===\n")

print("[ Python ]")
check("Version", lambda: sys.version.split()[0])

print("\n[ GPU / PyTorch ]")
import torch
check("PyTorch version", lambda: torch.__version__)
check("CUDA available", lambda: torch.cuda.is_available())
check("GPU name", lambda: torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
check("VRAM (GB)", lambda: f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}" if torch.cuda.is_available() else "N/A")

print("\n[ DRL Stack ]")
check("stable-baselines3", lambda: importlib.import_module("stable_baselines3").__version__)
check("sb3-contrib", lambda: importlib.import_module("sb3_contrib").__version__)
check("gymnasium", lambda: importlib.import_module("gymnasium").__version__)

print("\n[ Data ]")
check("yfinance", lambda: importlib.import_module("yfinance").__version__)
check("simfin", lambda: importlib.import_module("simfin").__version__)
check("pyarrow", lambda: importlib.import_module("pyarrow").__version__)
check("pandas", lambda: importlib.import_module("pandas").__version__)
check("numpy", lambda: importlib.import_module("numpy").__version__)

print("\n[ Features ]")
check("pandas-ta", lambda: "OK" if importlib.import_module("pandas_ta") else "?")
check("hmmlearn", lambda: importlib.import_module("hmmlearn").__version__)

print("\n[ Sentiment ]")
check("transformers", lambda: importlib.import_module("transformers").__version__)

print("\n[ Backtesting ]")
check("vectorbt", lambda: importlib.import_module("vectorbt").__version__)
check("numba", lambda: importlib.import_module("numba").__version__)

print("\n[ IBKR ]")
check("ib_async", lambda: "OK" if importlib.import_module("ib_async") else "?")

print("\n[ Config ]")
import yaml, pathlib
cfg_path = pathlib.Path("config/config.yaml")
uni_path = pathlib.Path("config/universe.yaml")
check("config.yaml exists", lambda: str(cfg_path) if cfg_path.exists() else (_ for _ in ()).throw(FileNotFoundError()))
check("universe.yaml exists", lambda: str(uni_path) if uni_path.exists() else (_ for _ in ()).throw(FileNotFoundError()))

with open(uni_path) as f:
    universe = yaml.safe_load(f)
total = sum(len(v["tickers"]) for v in universe["sectors"].values())
check("Universe stock count", lambda: total)

print("\n=== Verification complete ===")
