"""
One-shot setup and validation script for the paper trading computer.

Creates required directories, checks Python version, validates key
package imports, and tests IBKR Gateway connectivity.

Usage:
    python scripts/setup_other_computer.py
    python scripts/setup_other_computer.py --checkpoint-src /path/to/best_model.zip
"""
import sys
import socket
import argparse
import shutil
from pathlib import Path

REQUIRED_DIRS = [
    "logs",
    "results/live",
    "data/raw",
    "data/processed",
    "checkpoints/rppo_full_syn50/best",
]

REQUIRED_PACKAGES = [
    ("stable_baselines3", "stable-baselines3"),
    ("sb3_contrib",       "sb3-contrib"),
    ("gymnasium",         "gymnasium"),
    ("torch",             "torch"),
    ("numpy",             "numpy"),
    ("pandas",            "pandas"),
    ("pyarrow",           "pyarrow"),
    ("pandas_ta",         "pandas-ta"),
    ("hmmlearn",          "hmmlearn"),
    ("sklearn",           "scikit-learn"),
    ("ib_async",          "ib_async"),
    ("yfinance",          "yfinance"),
    ("yaml",              "pyyaml"),
    ("dotenv",            "python-dotenv"),
]

IBKR_PAPER_HOST = "127.0.0.1"
IBKR_PAPER_PORT = 7497


def check_python() -> bool:
    v = sys.version_info
    ok = v >= (3, 11)
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] Python {v.major}.{v.minor}.{v.micro}  (need 3.11+)")
    return ok


def create_dirs() -> bool:
    all_ok = True
    for d in REQUIRED_DIRS:
        p = Path(d)
        p.mkdir(parents=True, exist_ok=True)
        if p.exists():
            print(f"  [ OK] {d}/")
        else:
            print(f"  [FAIL] Could not create {d}/")
            all_ok = False
    return all_ok


def check_packages() -> bool:
    all_ok = True
    for module, pip_name in REQUIRED_PACKAGES:
        try:
            __import__(module)
            print(f"  [ OK] {pip_name}")
        except ImportError:
            print(f"  [FAIL] {pip_name}  — run: pip install {pip_name}")
            all_ok = False
    return all_ok


def check_checkpoint(src: str | None) -> bool:
    dst = Path("checkpoints/rppo_full_syn50/best/best_model.zip")

    if src:
        src_path = Path(src)
        if not src_path.exists():
            print(f"  [FAIL] Source checkpoint not found: {src}")
            return False
        shutil.copy2(src_path, dst)
        print(f"  [ OK] Checkpoint copied: {src} -> {dst}")
        return True

    if dst.exists():
        size_mb = dst.stat().st_size / 1_048_576
        print(f"  [ OK] Checkpoint present ({size_mb:.1f} MB): {dst}")
        return True

    print(f"  [WARN] Checkpoint not found: {dst}")
    print(f"         Copy it manually: checkpoints/rppo_full_syn50/best/best_model.zip")
    return False


def check_ibkr() -> bool:
    try:
        s = socket.create_connection((IBKR_PAPER_HOST, IBKR_PAPER_PORT), timeout=3)
        s.close()
        print(f"  [ OK] IBKR Gateway reachable at {IBKR_PAPER_HOST}:{IBKR_PAPER_PORT}")
        return True
    except (ConnectionRefusedError, OSError, socket.timeout) as e:
        print(f"  [WARN] IBKR Gateway not reachable: {e}")
        print( "         Start IB Gateway (paper mode, port 7497) before live trading.")
        return False


def check_config() -> bool:
    cfg = Path("config/config.yaml")
    env = Path(".env")
    ok  = True
    if cfg.exists():
        print(f"  [ OK] config/config.yaml")
    else:
        print(f"  [FAIL] config/config.yaml not found")
        ok = False
    if env.exists():
        print(f"  [ OK] .env")
    else:
        print(f"  [WARN] .env not found — copy .env.example and fill in API keys")
    return ok


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-src", default=None,
                   help="Source path of best_model.zip to copy into place")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n=== Setup check for paper trading computer ===\n")

    results = {}

    print("Python version:")
    results["python"] = check_python()

    print("\nDirectories:")
    results["dirs"] = create_dirs()

    print("\nPython packages:")
    results["packages"] = check_packages()

    print("\nCheckpoint:")
    results["checkpoint"] = check_checkpoint(args.checkpoint_src)

    print("\nConfig files:")
    results["config"] = check_config()

    print("\nIBKR Gateway connectivity:")
    results["ibkr"] = check_ibkr()

    # Summary
    passed = sum(1 for v in results.values() if v)
    total  = len(results)
    print(f"\n{'='*46}")
    print(f"  Result: {passed}/{total} checks passed")
    if passed == total:
        print("  GO — ready to run live trading.")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"  NO-GO — fix: {', '.join(failed)}")
    print(f"{'='*46}\n")

    print("Run command:")
    print("  python scripts/run_live_trading.py \\")
    print("      --checkpoint checkpoints/rppo_full_syn50/best/best_model.zip \\")
    print("      --log-file logs/live_trading.log\n")


if __name__ == "__main__":
    main()
