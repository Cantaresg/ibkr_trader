"""
Custom SB3 callbacks for the trading agent.

RegimeLoggingCallback  — logs current regime distribution to TensorBoard
EpisodeMetricsCallback — logs Sharpe, drawdown, NAV from info dicts
EarlyStopCallback      — stops training if eval Sharpe hasn't improved
"""
from __future__ import annotations
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


class EpisodeMetricsCallback(BaseCallback):
    """
    Collects per-step info dicts from all envs and logs episode-level metrics
    to TensorBoard at the end of each rollout.

    Metrics logged:
      train/episode_nav_final
      train/episode_drawdown_max
      train/episode_reward_mean
      train/regime_bull_mean, train/regime_bear_mean, train/regime_trans_mean
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._ep_rewards: list[float] = []
        self._ep_navs: list[float] = []
        self._ep_drawdowns: list[float] = []
        self._ep_regimes: list[np.ndarray] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "nav" in info:
                self._ep_navs.append(info["nav"])
            if "drawdown" in info:
                self._ep_drawdowns.append(info["drawdown"])
            if "reward" in info:
                self._ep_rewards.append(info["reward"])

        return True

    def _on_rollout_end(self) -> None:
        if not self._ep_rewards:
            return

        self.logger.record("train/episode_reward_mean", float(np.mean(self._ep_rewards)))
        if self._ep_navs:
            self.logger.record("train/episode_nav_final", float(np.mean(self._ep_navs[-len(self._ep_navs)//4 or -1:])))
        if self._ep_drawdowns:
            self.logger.record("train/episode_drawdown_max", float(np.max(self._ep_drawdowns)))

        self._ep_rewards.clear()
        self._ep_navs.clear()
        self._ep_drawdowns.clear()
        self._ep_regimes.clear()


class EarlyStopCallback(BaseCallback):
    """
    Stops training if eval Sharpe hasn't improved by `min_delta`
    for `patience` evaluations.

    Intended to be used together with EvalCallback — reads
    `self.parent.best_mean_reward` is not ideal; instead we track
    the custom `eval/sharpe` key logged by EvalTradingCallback.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.01, verbose: int = 0):
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self._best: float = -np.inf
        self._no_improve: int = 0

    def _on_step(self) -> bool:
        # This callback checks after each eval (called from EvalCallback.after_step)
        # We rely on the TensorBoard logger having written eval/sharpe
        sharpe = self.logger.name_to_value.get("eval/sharpe", None)
        if sharpe is None:
            return True

        if sharpe > self._best + self.min_delta:
            self._best = sharpe
            self._no_improve = 0
            if self.verbose:
                print(f"[EarlyStop] New best Sharpe: {sharpe:.4f}")
        else:
            self._no_improve += 1
            if self.verbose:
                print(f"[EarlyStop] No improvement {self._no_improve}/{self.patience}")
            if self._no_improve >= self.patience:
                print(f"[EarlyStop] Stopping: {self.patience} evals without improvement.")
                return False

        return True


class SharpeEvalCallback(EvalCallback):
    """
    EvalCallback that computes annualised Sharpe from actual daily NAV returns
    collected during eval rollouts via _log_success_callback, and logs it as
    eval/sharpe.  This replaces the old episode-reward-based approximation,
    which was contaminated by non-return reward components (diversification bonus,
    drawdown penalty) and used incorrect sqrt(252) scaling for episode-level rewards.
    """

    _BARS_PER_DAY: int = 7

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ep_nav_seq: list[float] = []   # NAV per bar in the current episode
        self._all_daily_rets: list[float] = []  # daily returns across all eval episodes

    def _log_success_callback(self, locals_: dict, globals_: dict) -> None:
        """Called by evaluate_policy at every env step during eval rollouts."""
        info = locals_.get("info", {})
        if isinstance(info, (list, tuple)):
            info = info[0] if info else {}

        nav = info.get("nav")
        if nav is not None:
            self._ep_nav_seq.append(float(nav))

        done = locals_.get("done", False)
        if isinstance(done, np.ndarray):
            done = bool(done.flat[0])
        elif isinstance(done, (list, tuple)):
            done = bool(done[0]) if done else False

        if done and len(self._ep_nav_seq) >= self._BARS_PER_DAY:
            navs = np.array(self._ep_nav_seq)
            # Day-end NAV = last bar of each trading day
            day_end_navs = navs[self._BARS_PER_DAY - 1::self._BARS_PER_DAY]
            if len(day_end_navs) > 1:
                daily_rets = np.diff(day_end_navs) / day_end_navs[:-1]
                self._all_daily_rets.extend(daily_rets.tolist())
            self._ep_nav_seq.clear()

    def _on_step(self) -> bool:
        # Clear accumulators before each potential eval run
        self._ep_nav_seq.clear()
        self._all_daily_rets.clear()

        result = super()._on_step()

        # Only log when we have enough daily returns from an actual eval run
        if len(self._all_daily_rets) > 5:
            rets = np.array(self._all_daily_rets, dtype=np.float64)
            std = rets.std()
            sharpe = (rets.mean() / std * np.sqrt(252)) if std > 1e-10 else 0.0
            self.logger.record("eval/sharpe", float(sharpe))

        return result
