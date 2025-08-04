#!/usr/bin/env bash
# Setup Python 3.11 and a virtual environment for the Colada app
set -e

# Step 1: Install system build dependencies
sudo apt update
sudo apt install -y build-essential zlib1g-dev libncurses5-dev \
  libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev \
  curl libbz2-dev wget libsqlite3-dev

# Step 2: Download and build Python 3.11 if not already installed or missing SQLite support
if ! command -v python3.11 >/dev/null 2>&1 || ! python3.11 -c "import sqlite3" >/dev/null 2>&1; then
  cd /usr/src
  sudo wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz
  sudo rm -rf Python-3.11.9
  sudo tar xzf Python-3.11.9.tgz
  cd Python-3.11.9
  sudo make clean
  sudo ./configure --enable-optimizations
  sudo make -j"$(nproc)"
  sudo make altinstall
fi

# Step 3: Verify Python 3.11 installation
python3.11 --version

# Step 4: Create and activate a virtual environment for the app
cd "$(dirname "$0")/.."
if [ ! -d venv ]; then
  python3.11 -m venv venv
fi
source venv/bin/activate

# Step 5: Upgrade pip and install Python packages using a custom TMPDIR
mkdir -p "$HOME/tmp"
TMPDIR="$HOME/tmp" pip install --upgrade pip setuptools wheel
TMPDIR="$HOME/tmp" pip install -r requirements.txt
