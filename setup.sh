#!/usr/bin/env bash
set -euo pipefail

# ─── Configuration ──────────────────────────────────────────────────
VENV_DIR=".venv"
MODEL_DIR="models"
# Place your GGUF model in models/ and update this filename:
MODEL_FILE="gemma-3n-E4B-it-Q4_K_M.gguf"

# ─── Create virtual environment ─────────────────────────────────────
echo "▸ Creating virtual environment in ${VENV_DIR}/ ..."
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# ─── Install dependencies ──────────────────────────────────────────
echo "▸ Installing Python dependencies ..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# ─── Verify model exists ───────────────────────────────────────────
mkdir -p "${MODEL_DIR}"
if [ ! -f "${MODEL_DIR}/${MODEL_FILE}" ]; then
    echo "⚠ Model not found at ${MODEL_DIR}/${MODEL_FILE}"
    echo "  Place your GGUF model file in ${MODEL_DIR}/ and update MODEL_FILE in setup.sh"
    exit 1
fi

# ─── Verify installation ───────────────────────────────────────────
echo "▸ Verifying installation ..."
python -c "import fastapi; import instructor; import qdrant_client; print('All imports OK')"
echo ""
echo "Setup complete. Activate the environment with:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Start the local LLM server with:"
echo "  python -m llama_cpp.server --model ${MODEL_DIR}/${MODEL_FILE} --port 8000"
