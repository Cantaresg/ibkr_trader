"""
Custom MLP feature extractor with LayerNorm for TradingEnv.

Used with SB3 MlpPolicy via:
    policy_kwargs = dict(
        features_extractor_class=TradingFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[512, 256],
        activation_fn=nn.ReLU,
    )
"""
import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.environment.trading_env import N_STOCKS, N_FEATURES, N_MARKET_FEATURES, PORTFOLIO_DIM


class TradingFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom extractor for the flat TradingEnv observation.

    Architecture:
      1. Split flat obs back into logical groups.
      2. stocks branch:  (N_STOCKS * lookback * N_FEATURES) → Linear → LayerNorm → ReLU
      3. market branch:  (lookback * N_MARKET_FEATURES)     → Linear → LayerNorm → ReLU
      4. portfolio head: (PORTFOLIO_DIM,)                   → Linear → LayerNorm → ReLU
      5. Concatenate all three → final Linear → LayerNorm → ReLU → features_dim
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 512,
        lookback: int = 30,
        stock_hidden: int = 256,
        market_hidden: int = 64,
        portfolio_hidden: int = 32,
    ):
        super().__init__(observation_space, features_dim)
        self.lookback = lookback

        self.stocks_dim    = N_STOCKS * lookback * N_FEATURES
        self.market_dim    = lookback * N_MARKET_FEATURES
        self.portfolio_dim = PORTFOLIO_DIM

        expected = self.stocks_dim + N_STOCKS + self.market_dim + self.portfolio_dim
        assert observation_space.shape[0] == expected, (
            f"Observation dim mismatch: expected {expected}, got {observation_space.shape[0]}"
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
        # Split flat observation into logical groups
        # Order matches FlattenDictObservation: stocks, stock_mask, market, portfolio
        s_end  = self.stocks_dim
        m_end  = s_end + N_STOCKS         # stock_mask (not used by extractor)
        mk_end = m_end + self.market_dim
        p_end  = mk_end + self.portfolio_dim

        stocks_flat = observations[:, :s_end]
        # stock_mask skipped (informative but very sparse; policy weights learn this implicitly)
        market_flat = observations[:, m_end:mk_end]
        portfolio   = observations[:, mk_end:p_end]

        s = self.stocks_branch(stocks_flat)
        m = self.market_branch(market_flat)
        p = self.portfolio_branch(portfolio)

        merged = torch.cat([s, m, p], dim=1)
        return self.merge(merged)
