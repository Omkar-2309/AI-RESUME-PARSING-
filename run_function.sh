#!/bin/bash
set -e

# Get the directory where the script is located
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$BASE_DIR/venv"

# Create a virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# Upgrade pip using the venv python explicitly
echo "Upgrading pip..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip

# Install dependencies using the venv pip explicitly
echo "Installing dependencies..."
"$VENV_DIR/bin/pip" install -r "$BASE_DIR/project/requirements.txt"

# Run the application using the venv python explicitly
echo "Starting application..."
cd "$BASE_DIR/project"
"$VENV_DIR/bin/python" app.py
