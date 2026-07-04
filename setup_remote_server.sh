#!/bin/bash
set -e

echo "=== STEP 0: Clean up old source lists ==="
rm -f /etc/apt/sources.list.d/pgdg.list /etc/apt/sources.list.d/timescaledb.list

echo "=== STEP 1: Add PostgreSQL and TimescaleDB APT Repositories ==="
apt-get update
apt-get install -y gnupg lsb-release wget apt-transport-https build-essential ninja-build git

# Import PostgreSQL GPG Key and add Repo
mkdir -p /etc/apt/keyrings
rm -f /etc/apt/keyrings/postgresql.gpg
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor -o /etc/apt/keyrings/postgresql.gpg

echo "deb [signed-by=/etc/apt/keyrings/postgresql.gpg] http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" | tee /etc/apt/sources.list.d/pgdg.list

# Import TimescaleDB GPG Key and add Repo
rm -f /etc/apt/keyrings/timescaledb.gpg
wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey | gpg --dearmor -o /etc/apt/keyrings/timescaledb.gpg
echo "deb [signed-by=/etc/apt/keyrings/timescaledb.gpg] https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/timescaledb.list

echo "=== STEP 2: Install System Packages ==="
apt-get update
# Install PostgreSQL 16, TimescaleDB, pgvector, Vulkan, and X11 development files
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  postgresql-16 \
  postgresql-contrib-16 \
  timescaledb-2-postgresql-16 \
  postgresql-16-pgvector \
  libvulkan-dev \
  libvulkan1 \
  libx11-dev

echo "=== STEP 3: Configure and Start PostgreSQL ==="
# Tune TimescaleDB (non-interactive)
timescaledb-tune --yes --quiet || echo "timescaledb-tune failed, skipping"

# Start the PostgreSQL service
service postgresql start

# Wait for database socket
echo "Waiting for PostgreSQL to start..."
for i in {1..30}; do
  if pg_isready -h 127.0.0.1 -p 5432; then
    echo "PostgreSQL is ready."
    break
  fi
  sleep 1
done

# Configure postgres user password
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'password';"

# Create the henri database
sudo -u postgres psql -c "CREATE DATABASE henri;" || echo "Database 'henri' may already exist."

# Enable extensions in henri database
sudo -u postgres psql -d henri -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"
sudo -u postgres psql -d henri -c "CREATE EXTENSION IF NOT EXISTS vector CASCADE;"

echo "=== STEP 4: Install Python Build Dependencies ==="
/venv/main/bin/python -m pip install --upgrade pip
/venv/main/bin/python -m pip install scikit-build-core nanobind cmake gguf scipy openai python-dotenv pydantic fastapi uvicorn psycopg[binary] h5py transformers

# Install llama-cpp-python compiled with CUDA acceleration
CMAKE_ARGS="-DGGML_CUDA=on -DLLAMA_CUDA=on" /venv/main/bin/python -m pip install llama-cpp-python --no-cache-dir

echo "=== STEP 5: Compile and Install wosx Bindings ==="
cd /workspace/HENRI/lib_physics
unzip -o wosx.zip
/venv/main/bin/python -m pip install ./wosx --config-settings=cmake.define.WOSX_ENABLE_GPU_SUPPORT=ON

echo "=== STEP 6: Initialize Database Schema ==="
export DATABASE_URL=postgresql://postgres:password@127.0.0.1:5432/henri
cd /workspace/HENRI
/venv/main/bin/python 6/init_local_db_4096.py

echo "=== STEP 7: Run HENRI Unified System Integrity Audits ==="
/venv/main/bin/python unify_system_integrity.py

echo "=== remote setup completed successfully! ==="
