# Base image with CUDA 12.1 and cuDNN 8 (Development headers required for llama-cpp-python compilation)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies, Python 3.10, and PostgreSQL
RUN apt-get update && apt-get install -y \
    software-properties-common \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    postgresql \
    postgresql-contrib \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up a virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3.10 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip and install core high-speed GPU dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install standard engine dependencies
RUN pip install --no-cache-dir \
    scikit-build-core \
    nanobind \
    cmake \
    gguf \
    scipy \
    openai \
    python-dotenv \
    pydantic \
    fastapi \
    uvicorn \
    psycopg[binary] \
    h5py \
    transformers \
    google-re2 \
    huggingface_hub

# Compile and install llama-cpp-python with strict CUDA acceleration
ENV CMAKE_ARGS="-DGGML_CUDA=on"
RUN pip install --no-cache-dir --verbose llama-cpp-python

# Copy the entire HENRI codebase into the container
COPY . /app/HENRI

# Create an entrypoint script to automatically boot the database and run the engine
RUN echo '#!/bin/bash\n\
service postgresql start\n\
# Wait for PostgreSQL to boot\n\
sleep 3\n\
# Setup database schema if it does not exist\n\
su - postgres -c "psql -c \\"CREATE DATABASE henri;\\" || true"\n\
su - postgres -c "psql -c \\"ALTER USER postgres WITH PASSWORD '\'password\'';\\""\n\
\n\
# Setup connection string\n\
export DATABASE_URL=postgresql://postgres:password@127.0.0.1:5432/henri\n\
export HF_TOKEN="${HF_TOKEN}"\n\
\n\
# Download Weights\n\
python -c "\n\
import os, sys\n\
from huggingface_hub import hf_hub_download\n\
token = os.environ.get('\''HF_TOKEN'\')\n\
if not os.path.exists('\''/app/HENRI/HENRI_CORE_V1/henri_fresh_core.pt'\'):\n\
    print('\''[*] Downloading weights from HuggingFace...'\'')\n\
    hf_hub_download(repo_id='\''Chandler/HENRI8.6Bswarm'\'', filename='\''henri_fresh_core.pt'\'', local_dir='\''/app/HENRI/HENRI_CORE_V1/'\'', token=token)\n\
"\n\
\n\
# Execute Benchmark Pipeline\n\
python /app/HENRI/HENRI_CORE_V1/unify_system_integrity.py\n\
python /app/HENRI/HENRI_CORE_V1/seed_universal_axioms.py\n\
exec python /app/HENRI/HENRI_CORE_V1/arc_live_benchmark.py\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Start the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
