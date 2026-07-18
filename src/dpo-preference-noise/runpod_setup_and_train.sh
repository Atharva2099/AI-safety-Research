set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="/workspace/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p /workspace/dpo-preference-noise
cd /workspace/dpo-preference-noise

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv venv .venv
uv pip install --python .venv/bin/python \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0
uv pip install --python .venv/bin/python \
    transformers datasets trl accelerate huggingface_hub

echo "Pod setup complete. Authenticate with: hf auth login"
echo "Then launch with: tmux new -s dpo10 'source .venv/bin/activate && python train_10_percent.py 2>&1 | tee train.log'"
