#!/bin/bash
# Shortcut to run the Streamlit vapp.py app

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v streamlit &> /dev/null; then
  echo "[run_vapp.sh] Error: streamlit is not installed. Please install it with 'pip install streamlit'"
  exit 1
fi

echo "[run_vapp.sh] Launching Streamlit app (vapp.py)..."
streamlit run vapp.py 