"""
Gymnasium API compliance tests for TradingEnv.
Run: python -m pytest tests/test_env_gym_api.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.environment.data_store import MarketDataStore
from src.environment.trading_env import TradingEnv, N_STOCKS, PORTFOLIO_DIM
from src.environment.wrappers import FlattenDictObservation


@pytest.fixture(scope="module")
def data_store():
    return MarketDataStore()


@pytest.fixture(scope="module")
def env(data_store):
    return TradingEnv(data_store, start_date="2015-01-01", end_date="2019-12-31", seed=42)


@pytest.fixture(scope="module")
def flat_env(data_store):
    base = TradingEnv(data_store, start_date="2015-01-01", end_date="2019-12-31", seed=42)
    return FlattenDictObservation(base)


class TestObservationSpace:
    def test_dict_obs_keys(self, env):
        obs, _ = env.reset(seed=0)
        assert set(obs.keys()) == {"stocks", "stock_mask", "market", "portfolio"}

    def test_obs_shapes(self, env):
        obs, _ = env.reset(seed=0)
        assert obs["stocks"].shape    == (N_STOCKS, 30, 33)
        assert obs["stock_mask"].shape == (N_STOCKS,)
        assert obs["market"].shape    == (30, 7)
        assert obs["portfolio"].shape == (PORTFOLIO_DIM,)

    def test_obs_dtype(self, env):
        obs, _ = env.reset(seed=0)
        for k, v in obs.items():
            assert v.dtype == np.float32, f"{k}: expected float32, got {v.dtype}"

    def test_stock_mask_binary(self, env):
        obs, _ = env.reset(seed=0)
        mask = obs["stock_mask"]
        assert set(mask).issubset({0.0, 1.0}), f"stock_mask has non-binary values: {np.unique(mask)}"

    def test_portfolio_sums_to_one(self, env):
        obs, _ = env.reset(seed=0)
        weights = obs["portfolio"][:N_STOCKS + 1]
        assert abs(weights.sum() - 1.0) < 1e-4, f"portfolio weights sum = {weights.sum():.6f}"

    def test_flat_obs_shape(self, flat_env):
        obs, _ = flat_env.reset(seed=0)
        expected = N_STOCKS * 30 * 33 + N_STOCKS + 30 * 7 + PORTFOLIO_DIM
        assert obs.shape == (expected,), f"flat obs shape {obs.shape} != ({expected},)"


class TestActionSpace:
    def test_action_shape(self, env):
        assert env.action_space.shape == (N_STOCKS + 1,)

    def test_random_action_in_space(self, env):
        for _ in range(10):
            a = env.action_space.sample()
            assert env.action_space.contains(a)


class TestEpisodeMechanics:
    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert "start_date" in info
        assert "tickers" in info
        assert len(info["tickers"]) == N_STOCKS

    def test_step_returns_correct_types(self, env):
        env.reset(seed=0)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_episode_terminates_after_252_steps(self, env):
        env.reset(seed=0)
        for i in range(252):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                assert i == 251, f"Episode ended at step {i}, expected 251"
                break
        else:
            pytest.fail("Episode did not terminate after 252 steps")

    def test_all_cash_action(self, env):
        """Agent putting everything in cash should produce near-zero return."""
        env.reset(seed=0)
        # Action that puts most weight on cash (last element)
        action = np.full(N_STOCKS + 1, -5.0, dtype=np.float32)
        action[-1] = 5.0  # cash logit very high
        _, reward, _, _, info = env.step(action)
        assert abs(info["portfolio_return"]) < 0.05, "All-cash should produce ~0 return"

    def test_portfolio_nav_positive(self, env):
        """NAV should stay positive throughout an episode."""
        env.reset(seed=1)
        for _ in range(50):
            _, _, _, truncated, info = env.step(env.action_space.sample())
            assert info["nav"] > 0, f"NAV went non-positive: {info['nav']}"
            if truncated:
                break


class TestGymnasiumCompliance:
    def test_check_env_flat(self, flat_env):
        """gymnasium.utils.env_checker.check_env must pass on the base Dict-obs env."""
        check_env(flat_env.unwrapped, warn=True, skip_render_check=True)
