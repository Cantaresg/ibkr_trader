"""Shared constants for the intraday DRL trading module."""

N_STOCKS         = 20   # agent observation slots (top-20 scanner output per day)
BARS_PER_DAY     = 7    # 9:30, 10:30, 11:30, 12:30, 13:30, 14:30, 15:30 ET
N_FEATURES       = 16   # per-stock intraday features
N_MARKET         = 5    # intraday market features
LOOKBACK         = 14   # bar lookback window (~2 trading days)

# portfolio state: stock_weights(20) + cash(1) + nav_norm(1) + drawdown(1)
PORTFOLIO_DIM    = N_STOCKS + 3   # = 23

# flat observation dim: stocks + mask + market + portfolio
OBS_DIM = N_STOCKS * LOOKBACK * N_FEATURES + N_STOCKS + LOOKBACK * N_MARKET + PORTFOLIO_DIM
# = 20*14*16 + 20 + 14*5 + 23 = 4480 + 20 + 70 + 23 = 4593

# Bar open hours in ET (integer hour of day when the bar opens)
BAR_HOURS_ET = [9, 10, 11, 12, 13, 14, 15]

# Path to shared universe definition (123 S&P 500 stocks across 6 sector buckets)
UNIVERSE_FILE = "config/universe.yaml"

# Fallback small universe used when universe.yaml is unavailable
INTRADAY_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "GOOGL", "TSLA", "AMD", "JPM", "BAC",
    "SPY", "QQQ",
]
