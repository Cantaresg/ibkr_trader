#!/bin/bash
# Server setup script for Hetzner CX33 (Ubuntu 22.04 x86)
# Run as the non-root user (ubuntu / whatever Hetzner creates)
# Usage: bash scripts/deploy_server.sh

set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo ""
echo "=================================================="
echo "  IBKR Trader — server setup"
echo "  $(date)"
echo "=================================================="

# ------------------------------------------------------------------
# 1. System packages
# ------------------------------------------------------------------
echo ""
echo "[1/6] Installing system packages..."
sudo apt-get update -q
sudo apt-get install -y \
    python3.11 python3.11-venv python3.11-distutils \
    openjdk-11-jre-headless \
    xvfb x11vnc \
    unzip curl wget git \
    htop

# ------------------------------------------------------------------
# 2. Python virtual environment
# ------------------------------------------------------------------
echo ""
echo "[2/6] Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3.11 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip setuptools wheel -q

# ------------------------------------------------------------------
# 3. Install Python packages
# ------------------------------------------------------------------
echo ""
echo "[3/6] Installing Python packages (CPU-only torch)..."
pip install torch -q  # ARM: use default PyPI index (no CUDA wheel needed)
pip install -r requirements_live.txt -q
echo "  Packages installed."

# ------------------------------------------------------------------
# 4. Create required directories
# ------------------------------------------------------------------
echo ""
echo "[4/6] Creating directories..."
mkdir -p logs results/live data/raw data/processed \
         checkpoints/rppo_full_syn50/best

# ------------------------------------------------------------------
# 5. Fix config paths (Google Drive -> local Linux paths)
# ------------------------------------------------------------------
echo ""
echo "[5/6] Patching config.yaml for Linux..."
sed -i 's|G:/My Drive/ibkr_news_raw|/home/$(whoami)/ibkr_news_raw|g' config/config.yaml
sed -i 's|G:/My Drive/ibkr_gdelt_raw|/home/$(whoami)/ibkr_gdelt_raw|g' config/config.yaml
echo "  config.yaml patched."

# ------------------------------------------------------------------
# 6. Create systemd service for the trader
# ------------------------------------------------------------------
echo ""
echo "[6/6] Creating systemd service..."
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
SERVICE_FILE="/etc/systemd/system/ibkr-trader.service"

sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=IBKR DRL Trader (rppo_full_syn50)
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$PROJECT_DIR
ExecStart=$VENV_PYTHON scripts/run_live_trading.py \\
    --checkpoint checkpoints/rppo_full_syn50/best/best_model.zip \\
    --log-file logs/live_trading.log
Restart=on-failure
RestartSec=60
StandardOutput=append:$PROJECT_DIR/logs/live_trading.log
StandardError=append:$PROJECT_DIR/logs/live_trading.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
echo "  Service created: ibkr-trader.service"
echo "  (do NOT start it yet — set up IB Gateway first)"

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo ""
echo "=================================================="
echo "  Setup complete. Next steps:"
echo ""
echo "  1. Upload checkpoint:"
echo "     checkpoints/rppo_full_syn50/best/best_model.zip"
echo ""
echo "  2. Create .env:"
echo "     cp .env.example .env && nano .env"
echo ""
echo "  3. Install IB Gateway + IBC (see docs/ibgateway_setup.md)"
echo ""
echo "  4. Validate:"
echo "     source .venv/bin/activate"
echo "     python scripts/setup_other_computer.py"
echo ""
echo "  5. Start trading:"
echo "     sudo systemctl start ibkr-trader"
echo "=================================================="
