"""
Train the intraday DRL trading agent.

Usage:
    python intraday_trader/scripts/train.py
    python intraday_trader/scripts/train.py --algo rppo --run-name intraday_rppo
    python intraday_trader/scripts/train.py --algo ppo  --synthetic-ratio 0.3 \\
                                            --synthetic-dir intraday_trader/data/processed/synthetic_episodes
    python intraday_trader/scripts/train.py --timesteps 500000 --run-name intraday_v1
    python intraday_trader/scripts/train.py --warm-start intraday_trader/checkpoints/best/best_model.zip
    python intraday_trader/scripts/train.py --train-start 2022-01-01 --train-end 2024-06-30 \\
                                            --eval-start 2024-07-01 --eval-end 2024-12-31
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pyarrow.parquet  # noqa: F401  — must precede torch/SB3 on Windows

from src.utils.config_loader import load_config
from src.utils.logging_config import setup_logging, get_logger
from intraday_trader.trainer import IntradayTrainer

log = get_logger("scripts.train_intraday")


def parse_args():
    p = argparse.ArgumentParser(description="Train the intraday DRL trader")
    p.add_argument("--config",      default="intraday_trader/config.yaml")
    p.add_argument("--algo",        default="ppo", choices=["ppo", "rppo"],
                   help="RL algorithm: ppo (default) or rppo (RecurrentPPO+LSTM)")
    p.add_argument("--run-name",    default=None,
                   help="Checkpoint sub-directory name (default: intraday_<algo>)")
    p.add_argument("--timesteps",   type=int,   default=None,
                   help="Override <algo>.total_timesteps")
    p.add_argument("--warm-start",  default=None,
                   help="Path to .zip checkpoint to warm-start from")
    p.add_argument("--train-start", default=None, help="Override data.start_date")
    p.add_argument("--train-end",   default=None, help="Override data.train_end")
    p.add_argument("--eval-start",  default=None, help="Override data.eval_start")
    p.add_argument("--eval-end",    default=None, help="Override data.eval_end")
    p.add_argument("--resume",      action="store_true",
                   help="Resume from latest checkpoint in the run's checkpoint dir")
    p.add_argument("--no-eval",     action="store_true", help="Skip final evaluation")
    p.add_argument("--lr",          type=float, default=None, help="Override <algo>.learning_rate")
    p.add_argument("--ent-coef",    type=float, default=None, help="Override <algo>.ent_coef")
    p.add_argument("--target-kl",   type=float, default=None, help="Override <algo>.target_kl (PPO/RPPO)")
    p.add_argument("--n-steps",     type=int,   default=None, help="Override <algo>.n_steps (PPO/RPPO)")
    p.add_argument("--n-days",      type=int,   default=None, help="Override environment.n_days_per_episode")
    p.add_argument("--lstm-hidden", type=int,   default=None,
                   help="Override rppo.lstm_hidden_size (RPPO only)")
    p.add_argument("--synthetic-ratio", type=float, default=None,
                   help="Fraction of bear episodes from synthetic pool (0.0–1.0)")
    p.add_argument("--synthetic-dir",   default=None,
                   help="Directory with .npz synthetic episodes (overrides config)")
    p.add_argument("--log-file",    default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)

    algo     = args.algo.lower()
    run_name = args.run_name or f"intraday_{algo}"

    cfg = load_config(args.config)

    data     = cfg.setdefault("data", {})
    algo_cfg = cfg.setdefault(algo, {})

    if args.train_start: data["start_date"] = args.train_start
    if args.train_end:   data["train_end"]  = args.train_end
    if args.eval_start:  data["eval_start"] = args.eval_start
    if args.eval_end:    data["eval_end"]   = args.eval_end

    if args.lr          is not None: algo_cfg["learning_rate"]   = args.lr
    if args.ent_coef    is not None: algo_cfg["ent_coef"]         = args.ent_coef
    if args.target_kl   is not None: algo_cfg["target_kl"]        = args.target_kl
    if args.n_steps     is not None: algo_cfg["n_steps"]          = args.n_steps
    if args.n_days      is not None: cfg.setdefault("environment", {})["n_days_per_episode"] = args.n_days
    if args.lstm_hidden is not None: algo_cfg["lstm_hidden_size"] = args.lstm_hidden

    trainer = IntradayTrainer(
        config          = cfg,
        config_path     = args.config,
        run_name        = run_name,
        warm_start_path = args.warm_start,
        algo            = algo,
        synthetic_ratio = args.synthetic_ratio,
        synthetic_dir   = args.synthetic_dir,
    )
    trainer.train(total_timesteps=args.timesteps, resume=args.resume)

    if not args.no_eval:
        results = trainer.evaluate(n_episodes=10)
        print(f"\nFinal evaluation ({algo.upper()}):")
        print(f"  Mean reward:    {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
        if "daily_sharpe" in results:
            print(f"  Daily Sharpe:   {results['daily_sharpe']:.3f}")
            print(f"  Bar Sharpe:     {results['bar_sharpe']:.3f}")
            print(f"  Max Drawdown:   {results['max_drawdown']:.1%}")
            print(f"  Daily Win Rate: {results['daily_win_rate']:.1%}")
            print(f"  Ann. Return:    {results['annualised_return']:.1%}")


if __name__ == "__main__":
    main()
