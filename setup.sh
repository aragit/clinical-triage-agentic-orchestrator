#!/usr/bin/env bash
set -euo pipefail

# ─── Configuration ──────────────────────────────────────────────────
VENV_DIR=".venv"
MODEL_DIR="models"
MODEL_FILE="Qwen2.5-3B-Instruct-Q4_K_M.gguf"
MODEL_REPO="Qwen/Qwen2.5-3B-Instruct-GGUF"

# ─── Create virtual environment ─────────────────────────────────────
echo "▸ Creating virtual environment in ${VENV_DIR}/ ..."
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# ─── Install dependencies ──────────────────────────────────────────
echo "▸ Installing Python dependencies ..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# ─── Download GGUF model ───────────────────────────────────────────
mkdir -p "${MODEL_DIR}"
if [ ! -f "${MODEL_DIR}/${MODEL_FILE}" ]; then
    echo "▸ Downloading ${MODEL_FILE} from HuggingFace ..."
    huggingface-cli download \
        "${MODEL_REPO}" \
        "${MODEL_FILE}" \
        --local-dir "${MODEL_DIR}"
    echo "▸ Model saved to ${MODEL_DIR}/${MODEL_FILE}"
else
    echo "▸ Model already exists at ${MODEL_DIR}/${MODEL_FILE}, skipping download."
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
