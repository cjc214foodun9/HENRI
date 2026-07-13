#!/bin/bash
set -e
echo "[SYSTEM] Initializing HENRI Thermodynamic Swarm Node..."

apt-get update && apt-get install -y git curl

WORKDIR="/opt/henri_swarm"
mkdir -p $WORKDIR
cd $WORKDIR

# Sync the codebase
git clone https://github.com/cjc214foodun9/HENRI.git || echo "Repo already exists"
cd HENRI
git submodule update --init --recursive

# Install missing python dependencies into the SYSTEM python 
# (Assuming you use the identical Vast.ai PyTorch image as your Master Node)
pip install asyncpg arc-agi

# Bind to the Master Engram Node
export POSTGRES_DSN="postgres://postgres:password@62.107.25.198:53468/henri"
export PYTHONPATH="$PYTHONPATH:$(pwd)/HENRI V2"
echo "[NETWORK] Master DSN Bound: $POSTGRES_DSN"

echo "[SYSTEM] Igniting Thermodynamic Inference Engine..."
nohup python3 arc_agi_3_thermodynamic_harness_client.py > worker_benchmark.log 2>&1 &
