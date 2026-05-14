"""
TradingEnv: gymnasium.Env for deep RL stock trading.

Observation space (Dict):
  stocks:     (n_stocks=20, lookback=30, n_features=33)  float32
  stock_mask: (n_stocks=20,)                              float32
  market:     (lookback=30, n_market_features=7)          float32
  portfolio:  (n_stocks+3+3 = 26,)                       float32
    = [stock_weights(20), cash_weight(1), nav_norm(1), drawdown(1), regime(3)]

Action space:
  Box(-1, 1, shape=(n_stocks+1,))  — raw logits; softmax applied internally
  Resulting weights are long-only and sum to 1.

Episode:
  - Start date sampled by regime then uniformly within regime (or uniformly if no weights given)
  - Top-20 scanner candidates selected at start date (fixed for the episode)
  - 252 steps (1 trading year)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from src.environment.data_store import MarketDataStore
from src.environment.portfolio import PortfolioState
from src.environment.reward import compute as compute_reward
from src.environment.synthetic_store import SyntheticEpisodeStore
from src.utils.logging_config import get_logger

log = get_logger("env.trading")

N_STOCKS         = 20
from src.features.pipeline import FEATURE_COLS              # noqa: E402
N_FEATURES       = len(FEATURE_COLS)
from src.features.market_features import N_MARKET_FEATURES  # noqa: E402
N_REGIME         = 3
PORTFOLIO_DIM    = N_STOCKS + 1 + 1 + 1 + N_REGIME  # weights + cash + nav + dd + regime = 26


class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_store: MarketDataStore,
        start_date: str = "2013-01-01",
        end_date: str   = "2020-12-31",
        lookback: int   = 30,
        episode_length: int = 252,
        initial_capital: float = 100_000.0,
        transaction_cost_bps: float = 10.0,
        reward_alpha: float = 1.0,
        reward_beta: float  = 2.0,
        reward_gamma: float = 0.5,
        drawdown_threshold: float = 0.05,
        regime_weights: dict[int, float] | None = None,
        synthetic_store: SyntheticEpisodeStore | None = None,
        synthetic_ratio: float = 0.0,
        seed: int | None = None,
    ):
        """
        regime_weights: target sampling probability per regime.
          Keys: 0=bull, 1=bear, 2=transition.
          Example: {0: 0.35, 1: 0.45, 2: 0.20} overweights bear episodes.
          None (default) samples uniformly across all valid start dates.
          Empty buckets are silently redistributed to remaining regimes.
        """
        super().__init__()
        self.ds             = data_store
        self.lookback       = lookback
        self.episode_length = episode_length
        self.initial_capital = initial_capital
        self.tc_bps         = transaction_cost_bps
        self.r_alpha        = reward_alpha
        self.r_beta         = reward_beta
        self.r_gamma        = reward_gamma
        self.dd_threshold   = drawdown_threshold

        self.valid_starts = data_store.valid_episode_start_indices(
            start_date, end_date, lookback, episode_length
        )
        if not self.valid_starts:
            raise ValueError(
                f"No valid episode start dates in [{start_date}, {end_date}]. "
                "Check data coverage and date range."
            )

        # Regime-balanced sampling setup
        self._regime_weights = regime_weights
        if regime_weights is not None:
            self._starts_by_regime = data_store.group_starts_by_regime(self.valid_starts)
            counts = {r: len(v) for r, v in self._starts_by_regime.items()}
            log.info("Regime buckets (bull/bear/trans): %s", counts)
        else:
            self._starts_by_regime = None

        # Synthetic episode support
        self._syn_store     = synthetic_store
        self._synthetic_ratio = float(synthetic_ratio)

        self.observation_space = spaces.Dict({
            "stocks":     spaces.Box(-4.0,  4.0, shape=(N_STOCKS, lookback, N_FEATURES), dtype=np.float32),
            "stock_mask": spaces.Box( 0.0,  1.0, shape=(N_STOCKS,),                     dtype=np.float32),
            "market":     spaces.Box(-4.0,  4.0, shape=(lookback, N_MARKET_FEATURES),    dtype=np.float32),
            "portfolio":  spaces.Box(-2.0,  2.0, shape=(PORTFOLIO_DIM,),                 dtype=np.float32),
        })

        self.action_space = spaces.Box(-1.0, 1.0, shape=(N_STOCKS + 1,), dtype=np.float32)

        self.portfolio = PortfolioState(N_STOCKS, initial_capital)

        # Episode state (set in reset)
        self._rng          = np.random.default_rng(seed)
        self._date_idx: int       = 0
        self._ticker_indices: list[int] = []
        self._step: int           = 0
        self._using_syn: bool     = False
        self._syn_ep: dict | None = None

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Sample start date — regime-balanced or uniform
        if self._regime_weights is not None and self._starts_by_regime is not None:
            available = [(r, self._starts_by_regime[r]) for r in [0, 1, 2]
                         if self._starts_by_regime[r]]
            regimes  = [r for r, _ in available]
            raw_w    = np.array([self._regime_weights.get(r, 0.0) for r in regimes], dtype=float)
            if raw_w.sum() == 0:
                raw_w = np.ones(len(regimes))
            probs = raw_w / raw_w.sum()
            chosen_regime = int(self._rng.choice(regimes, p=probs))
            self._date_idx = int(self._rng.choice(self._starts_by_regime[chosen_regime]))
        else:
            chosen_regime = None
            self._date_idx = int(self._rng.choice(self.valid_starts))

        # Decide whether to use a synthetic episode
        use_syn = (
            self._syn_store is not None
            and chosen_regime is not None
            and self._syn_store.has_regime(chosen_regime)
            and self._rng.random() < self._synthetic_ratio
        )
        if use_syn:
            self._syn_ep    = self._syn_store.sample(chosen_regime, self._rng)
            self._using_syn = True
        else:
            self._syn_ep    = None
            self._using_syn = False

        start_date = self.ds.dates[self._date_idx]

        # Select top-20 scanner candidates (fixed for this episode; also used as fallback)
        self._ticker_indices = self.ds.get_candidates(start_date, N_STOCKS)

        # Reset portfolio
        self.portfolio.reset()
        self._step = 0

        obs  = self._build_obs()
        info = {
            "start_date": str(start_date),
            "tickers":    self._selected_tickers(),
            "synthetic":  self._using_syn,
        }
        return obs, info

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        # 1. Convert raw action logits → portfolio weights via softmax
        weights = _softmax(action)  # shape (N_STOCKS + 1,)

        # 2. Compute stock returns for this day (from current date to next date)
        if self._using_syn and self._syn_ep is not None:
            prices_now  = self._syn_ep["close_prices"][:, self._step]
            prices_next = self._syn_ep["close_prices"][:, self._step + 1]
        else:
            next_date_idx = self._date_idx + 1
            prices_now  = self.ds.get_prices(self._date_idx,  self._live_ticker_indices())
            prices_next = self.ds.get_prices(next_date_idx, self._live_ticker_indices())

        stock_returns = np.zeros(N_STOCKS, dtype=np.float32)
        for j, (p_now, p_next) in enumerate(zip(prices_now, prices_next)):
            if p_now > 0 and p_next > 0:
                stock_returns[j] = (p_next - p_now) / p_now

        # 3. Update portfolio
        step_info = self.portfolio.step(weights, stock_returns, self.tc_bps)

        # 4. Compute reward
        reward, r_info = compute_reward(
            portfolio_return   = step_info["portfolio_return"],
            transaction_cost   = step_info["transaction_cost"],
            drawdown           = step_info["drawdown"],
            alpha              = self.r_alpha,
            beta               = self.r_beta,
            gamma              = self.r_gamma,
            drawdown_threshold = self.dd_threshold,
        )

        # 5. Advance date (only real-data episodes advance the calendar index)
        if not self._using_syn:
            self._date_idx += 1
        self._step += 1

        terminated = False
        truncated  = (self._step >= self.episode_length)

        obs  = self._build_obs()
        info = {**step_info, **r_info, "step": self._step,
                "date": str(self.ds.dates[self._date_idx])}
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self):
        date = self.ds.dates[self._date_idx]
        tickers = self._selected_tickers()
        stock_w = self.portfolio.weights[:N_STOCKS]
        cash_w  = self.portfolio.weights[-1]
        top3    = sorted(zip(tickers, stock_w), key=lambda x: -x[1])[:3]
        print(f"[{date.date()}] NAV={self.portfolio.nav:,.0f} | "
              f"Drawdown={self.portfolio.drawdown:.2%} | "
              f"Cash={cash_w:.1%} | "
              f"Top3={[(t, f'{w:.1%}') for t, w in top3]}")

    # ------------------------------------------------------------------
    def _build_obs(self) -> dict:
        if self._using_syn and self._syn_ep is not None:
            t = self._step
            stock_feat  = self._syn_ep["stock_features"][:, t: t + self.lookback, :]
            stock_mask  = self._syn_ep["stock_mask"]
            market_feat = self._syn_ep["market_features"][t: t + self.lookback, :]
            regime_prob = self._syn_ep["regime_probs"]
        else:
            stock_feat, stock_mask, market_feat, regime_prob = self.ds.get_obs_arrays(
                self._date_idx, self._ticker_indices, self.lookback
            )

        portfolio_state = self.portfolio.state_vector()  # (23,)
        portfolio_vec = np.concatenate([portfolio_state, regime_prob]).astype(np.float32)  # (26,)

        return {
            "stocks":     stock_feat,
            "stock_mask": stock_mask,
            "market":     market_feat,
            "portfolio":  portfolio_vec,
        }

    def _live_ticker_indices(self) -> list[int]:
        """Ticker indices, replacing -1 (padded) with 0 (harmless; prices will be 0)."""
        return [i if i >= 0 else 0 for i in self._ticker_indices]

    def _selected_tickers(self) -> list[str]:
        return [self.ds.ticker_list[i] if i >= 0 else '' for i in self._ticker_indices]


# ------------------------------------------------------------------
def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax → portfolio weights summing to 1, all ≥ 0."""
    x = np.asarray(x, dtype=np.float64) / temperature
    x = x - x.max()
    exp_x = np.exp(x)
    return (exp_x / exp_x.sum()).astype(np.float32)
