"""
3-state GaussianHMM regime detector.

States (auto-labeled after fitting):
  Bull   — low realised vol, high market breadth
  Bear   — high VIX change, low market breadth
  Trans  — everything else

Input features (6, subset of the 7 market features):
  vix_zscore, vix_term_structure, spy_trend_20d,
  market_breadth, yield_spread, credit_spread

Soft probabilities [P_bull, P_bear, P_trans] are stored per date.
HMM must be retrained on training-split data only for each walk-forward window.
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from hmmlearn import hmm

from src.utils.logging_config import get_logger

log = get_logger("regime.hmm")

# Columns used for HMM fitting (excludes put_call_ratio — zero-filled in Phase 1)
HMM_FEATURE_COLS = [
    "vix_zscore",
    "vix_term_structure",
    "spy_trend_20d",
    "market_breadth",
    "yield_spread",
    "credit_spread",
]
N_HMM_FEATURES = len(HMM_FEATURE_COLS)
N_STATES = 3


def fit(
    market_features: pd.DataFrame,
    n_states: int = N_STATES,
    n_iter: int = 200,
    seed: int = 42,
) -> tuple[hmm.GaussianHMM, np.ndarray]:
    """
    Fit GaussianHMM on market features.

    Returns:
        model    — fitted hmmlearn GaussianHMM
        labels   — integer array mapping state index -> [0=bull,1=bear,2=trans]
    """
    X = market_features[HMM_FEATURE_COLS].ffill().fillna(0).values
    lengths = [len(X)]  # one continuous sequence

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=seed,
        verbose=False,
    )
    model.fit(X, lengths)
    log.info("HMM fitted: %d states, converged=%s, score=%.1f",
             n_states, model.monitor_.converged, model.score(X, lengths))

    labels = _label_states(model, X)
    return model, labels


def _label_states(model: hmm.GaussianHMM, X: np.ndarray) -> np.ndarray:
    """
    Assign semantic labels to HMM states.

    Heuristic:
      Bull  = state with lowest mean spy_trend_20d std (lowest vol) + highest breadth
      Bear  = state with highest mean vix_zscore + lowest breadth
      Trans = remaining state

    Returns integer array of length n_states mapping state_idx -> label_idx
    (0=Bull, 1=Bear, 2=Trans).
    """
    means = model.means_  # shape (n_states, n_features)
    col_idx = {c: i for i, c in enumerate(HMM_FEATURE_COLS)}

    vix_means    = means[:, col_idx["vix_zscore"]]
    breadth_means = means[:, col_idx["market_breadth"]]

    # Bull: highest breadth (most stocks above 200d SMA)
    bull_state = int(np.argmax(breadth_means))
    # Bear: highest VIX z-score
    bear_state = int(np.argmax(vix_means))
    if bear_state == bull_state:
        # Resolve tie: pick the second highest VIX
        sorted_vix = np.argsort(vix_means)[::-1]
        bear_state = int(sorted_vix[1])
    # Transition: whichever state is neither bull nor bear
    all_states = {0, 1, 2}
    trans_state = list(all_states - {bull_state, bear_state})[0]

    labels = np.zeros(N_STATES, dtype=int)
    labels[bull_state]  = 0
    labels[bear_state]  = 1
    labels[trans_state] = 2

    log.info("State labels — Bull=%d, Bear=%d, Trans=%d",
             bull_state, bear_state, trans_state)
    return labels


def predict_proba(
    model: hmm.GaussianHMM,
    market_features: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Compute per-date soft regime probabilities [P_bull, P_bear, P_trans].
    Returns DataFrame with columns ['p_bull','p_bear','p_trans'].
    """
    X = market_features[HMM_FEATURE_COLS].ffill().fillna(0).values
    # posteriors: (n_dates, n_states)
    log_posteriors = model.predict_proba(X)  # hmmlearn returns probabilities, not log-probs

    # Reorder columns so index 0=bull, 1=bear, 2=trans
    reordered = np.zeros_like(log_posteriors)
    for state_idx, label in enumerate(labels):
        reordered[:, label] = log_posteriors[:, state_idx]

    return pd.DataFrame(
        reordered,
        index=market_features.index,
        columns=["p_bull", "p_bear", "p_trans"],
        dtype="float32",
    )


def save_model(model: hmm.GaussianHMM, labels: np.ndarray, window_id: str = "global") -> None:
    p = Path(f"data/processed/regime/hmm_{window_id}.pkl")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump({"model": model, "labels": labels}, f)
    log.info("HMM model saved: %s", p)


def load_model(window_id: str = "global") -> tuple[hmm.GaussianHMM, np.ndarray] | None:
    p = Path(f"data/processed/regime/hmm_{window_id}.pkl")
    if not p.exists():
        return None
    with open(p, "rb") as f:
        d = pickle.load(f)
    return d["model"], d["labels"]


def save_proba(proba: pd.DataFrame, window_id: str = "global") -> None:
    p = Path(f"data/processed/regime/proba_{window_id}.parquet")
    p.parent.mkdir(parents=True, exist_ok=True)
    proba.to_parquet(p)


def load_proba(window_id: str = "global") -> pd.DataFrame | None:
    p = Path(f"data/processed/regime/proba_{window_id}.parquet")
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    return df
