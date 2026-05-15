"""
IntradayTradingEnv: gymnasium.Env for intraday DRL stock trading.

Observation space (Dict):
  stocks:     (N_STOCKS=20, lookback=14, N_FEATURES=16)  float32
  stock_mask: (N_STOCKS=20,)                              float32
  market:     (lookback=14, N_MARKET=5)                  float32
  portfolio:  (PORTFOLIO_DIM=23,)                        float32
    = [stock_weights(20), cash_weight(1), nav_norm(1), drawdown(1)]

Action space:
  Box(-1, 1, shape=(N_STOCKS+1=21,)) — raw logits; softmax applied internally

Episode:
  - n_days_per_episode consecutive trading days (default 21 ≈ 1 month)
  - Each day = BARS_PER_DAY=7 hourly bars
  - At end-of-day bar (bar_idx == BARS_PER_DAY-1): portfolio forced flat before
    computing the next observation. The overnight exposure penalty is included
    in the reward for that bar.

Scanner integration:
  At reset() time, the environment queries data_store.get_candidates(date) to
  obtain the N_STOCKS tickers to trade in this episode. Tickers with index -1
  are masked (stock_mask=0, all-zero features).

Optional intraday scanner overlay:
    If enabled, ticker candidates are refreshed every N bars using lagged
    intraday rankings (no lookahead because rankings are shifted by one bar).

Synthetic episodes:
  If synthetic_store is provided and np.random.rand() < synthetic_ratio,
  the episode uses pre-generated negated returns instead of real data.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from intraday_trader.constants import (
    BARS_PER_DAY,
    LOOKBACK,
    N_FEATURES,
    N_MARKET,
    N_STOCKS,
)
from intraday_trader.data_store import IntradayDataStore
from intraday_trader.portfolio import IntradayPortfolioState
from intraday_trader.reward import compute as compute_reward
from src.utils.logging_config import get_logger

log = get_logger("intraday.env")


class IntradayTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_store: IntradayDataStore,
        start_date: str = "2022-01-01",
        end_date: str   = "2024-06-30",
        lookback: int   = LOOKBACK,
        n_stocks: int   = N_STOCKS,
        n_days_per_episode: int = 21,
        initial_capital: float = 5_000.0,
        transaction_cost_bps: float = 5.0,
        reward_alpha: float = 1.0,
        reward_beta: float  = 3.0,
        reward_gamma: float = 0.5,
        reward_delta: float = 0.001,
        reward_zeta: float  = 0.0,
        reward_eta: float   = 0.0,
        drawdown_threshold: float = 0.01,
        seed: int | None = None,
        synthetic_store=None,   # IntradaySyntheticStore | None
        synthetic_ratio: float = 0.0,
        intraday_scanner_enabled: bool = False,
        intraday_refresh_every_n_bars: int = 1,
        min_position_weight: float = 0.0,
        regime_balanced_sampling: bool = False,
        eod_force_flat: bool = True,
    ):
        super().__init__()
        self.ds              = data_store
        self.lookback        = lookback
        self.n_stocks        = n_stocks
        self.min_pos_weight  = float(min_position_weight)
        self.eod_force_flat  = eod_force_flat
        self.n_days          = n_days_per_episode
        self.initial_capital = initial_capital
        self.tc_bps          = transaction_cost_bps
        self.r_alpha         = reward_alpha
        self.r_beta          = reward_beta
        self.r_gamma         = reward_gamma
        self.r_delta         = reward_delta
        self.r_zeta          = reward_zeta
        self.r_eta           = reward_eta
        self.dd_threshold    = drawdown_threshold
        self.syn_store       = synthetic_store
        self.syn_ratio       = synthetic_ratio
        self.intraday_scanner_enabled = intraday_scanner_enabled
        self.intraday_refresh_every_n_bars = max(1, int(intraday_refresh_every_n_bars))

        self.valid_starts: list[int] = data_store.valid_episode_start_bars(
            start_date, end_date, lookback, n_days_per_episode
        )
        if not self.valid_starts:
            raise ValueError(
                f"No valid episode starts in [{start_date}, {end_date}]. "
                "Check intraday data coverage."
            )

        # Regime-balanced sampling weights (None = uniform)
        self._start_weights: np.ndarray | None = None
        if regime_balanced_sampling:
            self._start_weights = self._compute_regime_weights()
            n_bear = (self._start_weights > self._start_weights.mean()).sum()
            log.info(
                "Regime-balanced sampling: %d starts  bear~%d bull~%d neutral~%d",
                len(self.valid_starts),
                int(len(self.valid_starts) // 3),
                int(len(self.valid_starts) // 3),
                int(len(self.valid_starts) // 3),
            )

        portfolio_dim = n_stocks + 3
        self.observation_space = spaces.Dict({
            "stocks":     spaces.Box(-4.0, 4.0, shape=(n_stocks, lookback, N_FEATURES), dtype=np.float32),
            "stock_mask": spaces.Box( 0.0, 1.0, shape=(n_stocks,),                      dtype=np.float32),
            "market":     spaces.Box(-4.0, 4.0, shape=(lookback, N_MARKET),             dtype=np.float32),
            "portfolio":  spaces.Box(-2.0, 2.0, shape=(portfolio_dim,),                 dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_stocks + 1,), dtype=np.float32)

        self.portfolio = IntradayPortfolioState(n_stocks, initial_capital)
        self._rng      = np.random.default_rng(seed)
        self._start_bar_idx: int = 0
        self._flat_step: int     = 0

        # Current-episode ticker mapping: list of n_stocks indices into ds.tickers (-1 = masked)
        self._ticker_indices: list[int] = list(range(min(n_stocks, data_store.n_tickers)))
        while len(self._ticker_indices) < n_stocks:
            self._ticker_indices.append(-1)

        # Synthetic episode state
        self._using_syn: bool = False
        self._syn_ep: dict | None = None

    # ------------------------------------------------------------------
    def _compute_regime_weights(self) -> np.ndarray:
        """
        Equal-weight sampling across bear / neutral / bull regimes so the model
        trains on all market conditions in balanced proportion.

        Regime is determined by the 100-bar (≈14-day) rolling sum of the
        z-scored SPY bar return (market_arr column 0) ending at the episode
        start — a pure lookback signal with no lookahead.

        The bottom tercile of regime values is labelled bear, top tercile bull,
        middle neutral.  Sampling weights are scaled so each tercile contributes
        equally to the total expected draw regardless of how many calendar days
        fall in each regime.
        """
        spy_ret   = self.ds.market_arr[:, 0]  # z-scored spy bar return (float32)
        window    = 100
        cumsum    = np.cumsum(spy_ret.astype(np.float64))
        rolling   = np.empty_like(cumsum)
        rolling[:window] = cumsum[:window]
        rolling[window:] = cumsum[window:] - cumsum[:-window]

        regime_vals = np.array([rolling[i] for i in self.valid_starts], dtype=np.float64)

        p33 = np.percentile(regime_vals, 33.3)
        p67 = np.percentile(regime_vals, 66.7)
        is_bear    = regime_vals <= p33
        is_bull    = regime_vals >= p67
        is_neutral = ~(is_bear | is_bull)

        n = len(self.valid_starts)
        weights = np.ones(n, dtype=np.float64)
        for mask in (is_bear, is_neutral, is_bull):
            count = mask.sum()
            if count > 0:
                weights[mask] = (n / 3.0) / count

        total = weights.sum()
        return (weights / total).astype(np.float32)

    # ------------------------------------------------------------------
    @property
    def _total_steps(self) -> int:
        return self.n_days * BARS_PER_DAY

    @property
    def _current_flat_bar(self) -> int:
        return self._start_bar_idx + self._flat_step

    def _current_day_bar(self) -> tuple[int, int]:
        day_idx = self._flat_step // BARS_PER_DAY
        bar_idx = self._flat_step % BARS_PER_DAY
        return day_idx, bar_idx

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._start_bar_idx = int(self._rng.choice(self.valid_starts, p=self._start_weights))
        self._flat_step     = 0
        self.portfolio.reset()

        # --- Synthetic episode sampling ---
        self._using_syn = False
        self._syn_ep    = None
        if (
            self.syn_store is not None
            and self.syn_ratio > 0.0
            and self._rng.random() < self.syn_ratio
        ):
            ep = self.syn_store.sample(self._rng)
            if ep is not None:
                self._using_syn = True
                self._syn_ep    = ep

        # --- Scanner-based ticker selection ---
        start_ts = self.ds.bar_timestamps[self._start_bar_idx]
        self._ticker_indices = self.ds.get_candidates(
            pd.Timestamp(start_ts), n=self.n_stocks
        )
        if self.intraday_scanner_enabled and not self._using_syn:
            self._ticker_indices = self.ds.get_candidates_for_bar(
                self._start_bar_idx,
                n=self.n_stocks,
                use_intraday=True,
            )

        obs  = self._build_obs()
        info = {
            "start_bar":    self._start_bar_idx,
            "start_ts":     str(start_ts),
            "using_syn":    self._using_syn,
            "tickers":      [
                self.ds.tickers[i] if i >= 0 else "" for i in self._ticker_indices
            ],
        }
        return obs, info

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        _, bar_idx = self._current_day_bar()
        is_eod     = (bar_idx == BARS_PER_DAY - 1)

        # 1. Softmax action → portfolio weights
        if is_eod and self.eod_force_flat:
            # Override: force all-cash before computing returns on this bar
            weights = np.zeros(self.n_stocks + 1, dtype=np.float32)
            weights[-1] = 1.0
        else:
            weights = _softmax(action)
            weights = _apply_min_weight(weights, self.min_pos_weight, self.n_stocks)

        # 2. Overnight exposure BEFORE applying the forced flat (for reward penalty)
        overnight_exp = self.portfolio.overnight_exposure()

        # 3. Bar returns
        if self._using_syn and self._syn_ep is not None:
            bar_returns = self._get_syn_bar_returns()
        else:
            bar_returns = self._get_real_bar_returns()

        # 4. Portfolio step
        step_info = self.portfolio.step(weights, bar_returns, self.tc_bps)

        # 5. If EOD bar: force flat (if enabled), then record day PnL
        eod_tc = 0.0
        if is_eod:
            if self.eod_force_flat:
                eod_tc = self.portfolio.force_flat(self.tc_bps)
            self.portfolio.day_reset()

        # 6. Reward
        n_active = int((weights[:self.n_stocks] > 0).sum())
        reward, r_info = compute_reward(
            portfolio_return    = step_info["portfolio_return"],
            transaction_cost    = step_info["transaction_cost"] + eod_tc,
            drawdown            = step_info["drawdown"],
            overnight_exposure  = overnight_exp,
            is_eod_bar          = is_eod,
            n_active_positions  = n_active,
            n_total_stocks      = self.n_stocks,
            alpha               = self.r_alpha,
            beta                = self.r_beta,
            gamma               = self.r_gamma,
            delta               = self.r_delta,
            zeta                = self.r_zeta,
            eta                 = self.r_eta,
            drawdown_threshold  = self.dd_threshold,
        )

        # 7. Advance
        self._flat_step += 1
        terminated = False
        truncated  = (self._flat_step >= self._total_steps)

        # Optional bar-level candidate refresh for live intraday screener.
        if (
            not truncated
            and not self._using_syn
            and self.intraday_scanner_enabled
            and (self._flat_step % self.intraday_refresh_every_n_bars == 0)
        ):
            self._ticker_indices = self.ds.get_candidates_for_bar(
                self._current_flat_bar,
                n=self.n_stocks,
                use_intraday=True,
            )

        obs  = self._build_obs()
        info = {
            **step_info, **r_info,
            "flat_step":          self._flat_step,
            "bar_idx":            bar_idx,
            "is_eod":             is_eod,
            "n_active_positions": int((weights[:self.n_stocks] > 0).sum()),
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self):
        _, bar_idx = self._current_day_bar()
        bar_ts  = self.ds.bar_timestamps[self._current_flat_bar]
        stock_w = self.portfolio.weights[:self.n_stocks]
        cash_w  = self.portfolio.weights[-1]
        print(
            f"[{bar_ts}] bar={bar_idx} NAV={self.portfolio.nav:,.0f} "
            f"DD={self.portfolio.drawdown:.2%} Cash={cash_w:.1%} "
            f"StockExp={stock_w.sum():.1%}"
        )

    # ------------------------------------------------------------------
    def _build_obs(self) -> dict:
        if self._using_syn and self._syn_ep is not None:
            return self._build_syn_obs()
        return self._build_real_obs()

    def _build_real_obs(self) -> dict:
        bar = self._current_flat_bar
        bar = min(bar, self.ds.n_bars - 1)

        # Get features for the selected tickers in this episode
        stock_feat, stock_mask, market_feat = self.ds.get_obs_arrays_for_tickers(
            bar, self._ticker_indices, self.lookback
        )
        portfolio_vec = self.portfolio.state_vector()

        return {
            "stocks":     stock_feat,
            "stock_mask": stock_mask,
            "market":     market_feat,
            "portfolio":  portfolio_vec,
        }

    def _build_syn_obs(self) -> dict:
        """Build observation from synthetic episode data."""
        ep   = self._syn_ep
        step = min(self._flat_step, ep["stock_features"].shape[1] - 1)

        # stock_features shape: (n_stocks, total_bars, N_FEATURES)
        start = max(0, step - self.lookback + 1)
        end   = step + 1
        wlen  = end - start

        stock_feat = np.zeros((self.n_stocks, self.lookback, N_FEATURES), dtype=np.float32)
        n = min(self.n_stocks, ep["stock_features"].shape[0])
        stock_feat[:n, self.lookback - wlen:, :] = ep["stock_features"][:n, start:end, :]

        stock_mask = ep.get("stock_mask", np.ones(self.n_stocks, dtype=np.float32))
        if len(stock_mask) < self.n_stocks:
            m = np.zeros(self.n_stocks, dtype=np.float32)
            m[:len(stock_mask)] = stock_mask
            stock_mask = m

        market_feat = np.zeros((self.lookback, N_MARKET), dtype=np.float32)
        # Use real market features for the same bars (macro context unchanged)
        bar = min(self._current_flat_bar, self.ds.n_bars - 1)
        market_feat[self.lookback - wlen:, :] = self.ds.market_arr[bar - wlen + 1:bar + 1, :]

        portfolio_vec = self.portfolio.state_vector()
        return {
            "stocks":     stock_feat,
            "stock_mask": stock_mask,
            "market":     market_feat,
            "portfolio":  portfolio_vec,
        }

    def _get_real_bar_returns(self) -> np.ndarray:
        """Bar returns for the current episode's selected tickers."""
        flat_bar = self._current_flat_bar
        next_idx = flat_bar + 1
        if next_idx >= self.ds.n_bars:
            return np.zeros(self.n_stocks, dtype=np.float32)

        rets = np.zeros(self.n_stocks, dtype=np.float32)
        for slot, ti in enumerate(self._ticker_indices):
            if ti < 0:
                continue
            p_now  = self.ds.close_arr[flat_bar,  ti]
            p_next = self.ds.close_arr[next_idx,   ti]
            if p_now > 0 and p_next > 0:
                rets[slot] = (p_next - p_now) / p_now
        return rets

    def _get_syn_bar_returns(self) -> np.ndarray:
        """Bar returns from synthetic episode close prices."""
        ep   = self._syn_ep
        step = self._flat_step
        # close_prices shape: (n_stocks, total_bars + 1)
        close = ep.get("close_prices")
        if close is None or step >= close.shape[1] - 1:
            return np.zeros(self.n_stocks, dtype=np.float32)

        rets = np.zeros(self.n_stocks, dtype=np.float32)
        n = min(self.n_stocks, close.shape[0])
        p_now  = close[:n, step]
        p_next = close[:n, step + 1]
        mask   = (p_now > 0) & (p_next > 0)
        rets[:n] = np.where(mask, (p_next - p_now) / p_now, 0.0)
        return rets


# ------------------------------------------------------------------
def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64) / temperature
    x = x - x.max()
    exp_x = np.exp(x)
    return (exp_x / exp_x.sum()).astype(np.float32)


def _apply_min_weight(weights: np.ndarray, threshold: float, n_stocks: int) -> np.ndarray:
    """Zero stock slots below `threshold`; freed weight flows to cash via renormalisation."""
    if threshold <= 0.0:
        return weights
    w = weights.copy()
    w[:n_stocks][w[:n_stocks] < threshold] = 0.0
    total = w.sum()
    if total > 1e-8:
        w /= total
    else:
        w[:] = 0.0
        w[n_stocks] = 1.0
    return w.astype(np.float32)


# Lazy import to avoid circular at module level
import pandas as pd  # noqa: E402 — placed after class definition intentionally
