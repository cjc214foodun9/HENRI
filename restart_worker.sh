cd /opt/henri_swarm/HENRI
source venv/bin/activate
export LD_PRELOAD=/opt/henri_swarm/HENRI/venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
export PYTHONPATH="$PYTHONPATH:$(pwd)/HENRI V2"
export POSTGRES_DSN="postgres://postgres:password@62.107.25.198:53468/henri"
nohup python3 arc_agi_3_thermodynamic_harness_client.py > worker_benchmark.log 2>&1 &
