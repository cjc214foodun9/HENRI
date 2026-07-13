\#\!/bin/bash

# **Aletheia \- Bare-Metal Swarm Node Provisioning Script**

# **Executes deterministic environment initialization for HENRI worker instances.**

set \-e

# **STREAMING\_CHUNK:Initializing system dependencies...**

echo "\[SYSTEM\] Initializing HENRI Thermodynamic Swarm Node..."

# **1\. System Updates and Base Dependencies**

# **Enforcing a strict apt update to prevent package collision on fresh Vast.ai images**

apt-get update && apt-get install \-y git python3-pip python3-venv curl

# **2\. Establish Isolated Workspace**

WORKDIR="/opt/henri\_swarm"

mkdir \-p $WORKDIR

cd $WORKDIR

# **STREAMING\_CHUNK:Cloning the continuous-time codebase...**

# **3\. Clone Repository**

# **(Note: In your actual deployment, replace with your authenticated repo URL if private)**

REPO\_URL="https://github.com/your-org/henri-core.git"

if \[ \! \-d "henri-core" \]; then

git clone $REPO\_URL

fi

cd henri-core

# **STREAMING\_CHUNK:Enforcing the exact mathematical dependency matrix...**

# **4\. Enforce Exact Dependency Matrix**

# **We isolate the python environment to prevent host OS library corruption**

python3 \-m venv venv

source venv/bin/activate

pip install \--upgrade pip

# **Installing the strict mathematical and async substrates**

pip install torch numpy asyncpg

# **STREAMING\_CHUNK:Binding to the Master Engram Node...**

# **5\. Bind to the Master Engram Node**

# **The public IP and mapped port of the Master Vast.ai node**

export POSTGRES\_DSN="postgres://postgres:password@62.107.25.198:53468/henri"

echo "\[NETWORK\] Master DSN Bound: $POSTGRES\_DSN"

# **STREAMING\_CHUNK:Igniting the thermodynamic inference loop...**

# **6\. Ignite the Continuous-Time Inference Loop**

echo "\[SYSTEM\] Igniting Thermodynamic Inference Engine..."

# **The script will now run autonomously, pulling from and writing to Zone C**

python3 arc\_agi\_3\_thermodynamic\_harness\_client.py