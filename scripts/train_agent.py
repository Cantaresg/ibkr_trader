"""
Train the DRL trading agent.

Usage:
    python scripts/train_agent.py --algo ppo
    python scripts/train_agent.py --algo sac  --run-name sac_best --timesteps 2000000
    python scripts/train_agent.py --algo rppo --run-name rppo_best --timesteps 2000000
    python scripts/train_agent.py --timesteps 2000000 --run-name phase1_baseline
    python scripts/train_agent.py --warm-start checkpoints/phase1_mlp/ppo_2000000_steps.zip
    python scripts/train_agent.py --train-start 2013-01-01 --train-end 2018-12-31 \\
                                  --eval-start 2019-01-01 --eval-end 2019-12-31
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Must be imported before torch/SB3 to avoid Windows DLL conflict with CUDA
import pyarrow.parquet  # noqa: F401

from src.utils.config_loader import load_config
from src.utils.logging_config import setup_logging
from src.training.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",        default="config/config.yaml")
    p.add_argument("--algo",          default="ppo", choices=["ppo", "sac", "rppo"],
                   help="RL algorithm: ppo (default), sac, rppo (recurrent PPO with LSTM)")
    p.add_argument("--run-name",      default="phase1_mlp")
    p.add_argument("--timesteps",     type=int,   default=None,
                   help="Override <algo>.total_timesteps from config")
    p.add_argument("--warm-start",    default=None,
                   help="Path to a .zip checkpoint to warm-start from")
    p.add_argument("--train-start",   default="2013-01-01")
    p.add_argument("--train-end",     default="2018-12-31")
    p.add_argument("--eval-start",    default="2019-01-01")
    p.add_argument("--eval-end",      default="2019-12-31")
    p.add_argument("--no-eval",       action="store_true",
                   help="Skip final evaluation after training")
    # Per-experiment overrides — each changes exactly one hyperparameter
    # PPO
    p.add_argument("--target-kl",     type=float, default=None, help="Override ppo.target_kl")
    p.add_argument("--ent-coef",      type=float, default=None, help="Override ppo.ent_coef")
    p.add_argument("--lr",            type=float, default=None, help="Override ppo.learning_rate")
    p.add_argument("--n-steps",       type=int,   default=None, help="Override ppo.n_steps")
    p.add_argument("--gamma",         type=float, default=None, help="Override ppo.gamma")
    p.add_argument("--clip-range",    type=float, default=None, help="Override ppo.clip_range")
    p.add_argument("--batch-size",    type=int,   default=None, help="Override ppo.batch_size")
    p.add_argument("--vf-coef",       type=float, default=None, help="Override ppo.vf_coef")
    # Reward
    p.add_argument("--drawdown-beta",      type=float, default=None, help="Override reward.drawdown_penalty_weight")
    p.add_argument("--drawdown-threshold", type=float, default=None, help="Override reward.drawdown_threshold")
    # Environment
    p.add_argument("--episode-length", type=int, default=None, help="Override environment.episode_length")
    # Model architecture
    p.add_argument("--features-dim",   type=int, default=None, help="Override model.features_dim (extractor output)")
    # Regime-balanced episode sampling
    p.add_argument("--regime-weights", default=None,
                   help="Comma-separated bull,bear,trans weights e.g. '0.35,0.45,0.20'. "
                        "None = uniform sampling (default).")
    # Synthetic bear episode augmentation
    p.add_argument("--synthetic-dir",   default=None,
                   help="Path to pre-generated synthetic .npz episodes directory")
    p.add_argument("--synthetic-ratio", type=float, default=0.0,
                   help="Fraction of bear episodes drawn from synthetic pool (0.0–1.0)")
    # Logging
    p.add_argument("--log-file",       default=None, help="Append logs to this file in addition to stdout")
    p.add_argument("--resume",         action="store_true",
                   help="Continue step counter from checkpoint (use with --warm-start). "
                        "--timesteps then means *remaining* steps, not total.")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)

    cfg = load_config(args.config)
    cfg["config_path"]    = args.config
    cfg["training_start"] = args.train_start
    cfg["training_end"]   = args.train_end
    cfg["eval_start"]     = args.eval_start
    cfg["eval_end"]       = args.eval_end

    # Apply single-param overrides before handing cfg to Trainer
    ppo = cfg["ppo"]
    if args.target_kl     is not None: ppo["target_kl"]     = args.target_kl
    if args.ent_coef      is not None: ppo["ent_coef"]      = args.ent_coef
    if args.lr            is not None: ppo["learning_rate"] = args.lr
    if args.n_steps       is not None: ppo["n_steps"]       = args.n_steps
    if args.gamma         is not None: ppo["gamma"]         = args.gamma
    if args.clip_range    is not None: ppo["clip_range"]    = args.clip_range
    if args.batch_size    is not None: ppo["batch_size"]    = args.batch_size
    if args.vf_coef       is not None: ppo["vf_coef"]       = args.vf_coef

    rwd = cfg["reward"]
    if args.drawdown_beta      is not None: rwd["drawdown_penalty_weight"] = args.drawdown_beta
    if args.drawdown_threshold is not None: rwd["drawdown_threshold"]      = args.drawdown_threshold

    if args.episode_length is not None:
        cfg["environment"]["episode_length"] = args.episode_length
    if args.features_dim is not None:
        cfg.setdefault("model", {})["features_dim"] = args.features_dim

    if args.regime_weights is not None:
        parts = [float(x) for x in args.regime_weights.split(",")]
        cfg["regime_weights"] = {0: parts[0], 1: parts[1], 2: parts[2]}

    if args.synthetic_dir is not None:
        cfg["synthetic_dir"] = args.synthetic_dir
    if args.synthetic_ratio > 0.0:
        cfg["synthetic_ratio"] = args.synthetic_ratio

    trainer = Trainer(cfg, run_name=args.run_name, warm_start_path=args.warm_start, algo=args.algo)
    trainer.train(total_timesteps=args.timesteps, reset_num_timesteps=not args.resume)

    if not args.no_eval:
        results = trainer.evaluate(n_episodes=20)
        print(f"\nFinal evaluation:")
        print(f"  Mean reward: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")


if __name__ == "__main__":
    main()
