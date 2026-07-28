#!/bin/bash
cd '/workspace/HENRI V2/HENRI V2'
mkdir -p logs
export ZONE_C_ENV=prod
export ZONE_C_PROD_DSN='postgresql://postgres:postgres@localhost:10100/henri'
export CONSTRAINT_AXIOM=1
export PROGRESS_VALENCE=1
export LEARNABLE_ACTIONS=1
export HAPPY_TENSOR_CUT=1
export LAMBDA_GOAL=1.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.
exec /venv/main/bin/python production_arc_run.py > logs/production_run.log 2>&1
