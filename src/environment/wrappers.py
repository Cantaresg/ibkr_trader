"""Wrappers for TradingEnv."""
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box


class FlattenDictObservation(gym.ObservationWrapper):
    """
    Flattens the Dict observation space into a single 1D Box.
    Used for Phase 1 MLP policy.
    Order: stocks (flattened) + stock_mask + market (flattened) + portfolio
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        orig = env.observation_space
        self._keys_order = ["stocks", "stock_mask", "market", "portfolio"]
        flat_dim = sum(
            int(np.prod(orig[k].shape)) for k in self._keys_order
        )
        self.observation_space = Box(
            low=-4.0, high=4.0, shape=(flat_dim,), dtype=np.float32
        )

    def observation(self, obs: dict) -> np.ndarray:
        parts = [obs[k].flatten() for k in self._keys_order]
        return np.concatenate(parts).astype(np.float32)
