"""
Custom MLP feature extractor for IntradayTradingEnv.

Sized for the 4593-dim intraday observation (N_STOCKS=20):
  stocks branch:   (20 * 14 * 16 = 4480) → Linear(512) → LayerNorm → ReLU
  market branch:   (14 * 5 = 70)         → Linear(32)  → LayerNorm → ReLU
  portfolio branch:(23,)                  → Linear(16)  → LayerNorm → ReLU
  merge:           (560,)                 → Linear(256) → LayerNorm → ReLU

Used with SB3 MlpPolicy via:
    policy_kwargs = dict(
        features_extractor_class=IntradayFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[256, 128],
        activation_fn=nn.ReLU,
    )
"""
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from intraday_trader.constants import (
    LOOKBACK,
    N_FEATURES,
    N_MARKET,
    N_STOCKS,
)


class IntradayFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom extractor for the flat IntradayTradingEnv observation.
    Order matches FlattenDictObservation: stocks, stock_mask, market, portfolio.
    stock_mask is skipped (policy weights learn valid slots implicitly).
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int  = 256,
        n_stocks: int      = N_STOCKS,
        lookback: int      = LOOKBACK,
        stock_hidden: int  = 512,
        market_hidden: int = 32,
        portfolio_hidden: int = 16,
    ):
        super().__init__(observation_space, features_dim)
        self.n_stocks  = n_stocks
        self.lookback  = lookback

        self.stocks_dim    = n_stocks * lookback * N_FEATURES
        self.market_dim    = lookback * N_MARKET
        self.portfolio_dim = n_stocks + 3

        expected = self.stocks_dim + n_stocks + self.market_dim + self.portfolio_dim
        assert observation_space.shape[0] == expected, (
            f"Obs dim mismatch: expected {expected}, got {observation_space.shape[0]}"
        )

        self.stocks_branch = nn.Sequential(
            nn.Linear(self.stocks_dim, stock_hidden),
            nn.LayerNorm(stock_hidden),
            nn.ReLU(),
        )
        self.market_branch = nn.Sequential(
            nn.Linear(self.market_dim, market_hidden),
            nn.LayerNorm(market_hidden),
            nn.ReLU(),
        )
        self.portfolio_branch = nn.Sequential(
            nn.Linear(self.portfolio_dim, portfolio_hidden),
            nn.LayerNorm(portfolio_hidden),
            nn.ReLU(),
        )

        merged_dim = stock_hidden + market_hidden + portfolio_hidden
        self.merge = nn.Sequential(
            nn.Linear(merged_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        s_end  = self.stocks_dim
        m_end  = s_end + self.n_stocks     # skip stock_mask
        mk_end = m_end + self.market_dim
        p_end  = mk_end + self.portfolio_dim

        stocks_flat = observations[:, :s_end]
        market_flat = observations[:, m_end:mk_end]
        portfolio   = observations[:, mk_end:p_end]

        s = self.stocks_branch(stocks_flat)
        m = self.market_branch(market_flat)
        p = self.portfolio_branch(portfolio)

        merged = torch.cat([s, m, p], dim=1)
        return self.merge(merged)
