#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# SpamShield — Environment Setup for RTX 5090 (Blackwell)
# Author: Ricky (2702243016)
# ═══════════════════════════════════════════════════════════════════

set -e  # Exit on any error

ENV_NAME="spamshield"
PYTHON_VERSION="3.11"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   SpamShield Environment Setup — RTX 5090           ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Check NVIDIA driver ───────────────────────────────────
echo "[1/7] Checking NVIDIA driver..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "  ❌ nvidia-smi not found. Install NVIDIA driver 560+ first."
    exit 1
fi

DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "  ✅ Driver: $DRIVER"

CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}')
echo "  ✅ CUDA Version (driver): $CUDA_VER"

# Verify CUDA >= 12.6 for Blackwell (sm_120)
CUDA_MAJOR=$(echo $CUDA_VER | cut -d. -f1)
CUDA_MINOR=$(echo $CUDA_VER | cut -d. -f2)
if [ "$CUDA_MAJOR" -lt 12 ] || ([ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -lt 6 ]); then
    echo ""
    echo "  ⚠️  WARNING: CUDA $CUDA_VER detected."
    echo "      RTX 5090 (Blackwell) requires CUDA 12.6+ for full support."
    echo "      Recommended: CUDA 12.8"
    echo "      Update driver from: https://www.nvidia.com/drivers"
    echo ""
    read -p "  Continue anyway? (y/N): " choice
    [[ "$choice" != "y" && "$choice" != "Y" ]] && exit 1
fi

# ── Step 2: Check/Install conda ───────────────────────────────────
echo ""
echo "[2/7] Checking conda..."
if ! command -v conda &> /dev/null; then
    echo "  ❌ conda not found."
    echo "  Install Miniconda from: https://docs.conda.io/en/latest/miniconda.html"
    echo "  Quick install:"
    echo "    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "    bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi
echo "  ✅ conda found: $(conda --version)"

# ── Step 3: Create conda env ──────────────────────────────────────
echo ""
echo "[3/7] Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."

if conda info --envs | grep -q "^$ENV_NAME "; then
    echo "  ⚠️  Environment '$ENV_NAME' already exists."
    read -p "  Remove and recreate? (y/N): " choice
    if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
        conda deactivate 2>/dev/null || true
        conda remove -n $ENV_NAME --all -y
    else
        echo "  Skipping environment creation."
    fi
fi

conda create -n $ENV_NAME python=$PYTHON_VERSION -y
echo "  ✅ Environment '$ENV_NAME' created"

# ── Step 4: Activate env ─────────────────────────────────────────
echo ""
echo "[4/7] Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME
echo "  ✅ Activated: $CONDA_DEFAULT_ENV"

# ── Step 5: Install PyTorch (CUDA 12.6) ──────────────────────────
echo ""
echo "[5/7] Installing PyTorch with CUDA 12.6..."
echo "  (This may take several minutes — ~2.5GB download)"

pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126 \
    --quiet

# Verify installation
python -c "
import torch
print(f'  ✅ PyTorch     : {torch.__version__}')
print(f'  ✅ CUDA avail  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  ✅ GPU         : {torch.cuda.get_device_name(0)}')
    print(f'  ✅ VRAM        : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
    print(f'  ✅ CUDA vers   : {torch.version.cuda}')
    # Test BF16 (critical for Blackwell)
    x = torch.randn(4, 4, dtype=torch.bfloat16, device=\"cuda\")
    print(f'  ✅ BF16 test   : OK (Blackwell native)')
"

# ── Step 6: Install all packages ─────────────────────────────────
echo ""
echo "[6/7] Installing dependencies..."

pip install --quiet \
    transformers>=4.40.0 \
    datasets \
    tokenizers \
    accelerate \
    optuna>=3.6.0 \
    optuna-dashboard \
    plotly \
    scikit-learn \
    pandas \
    numpy \
    scipy \
    matplotlib \
    seaborn \
    tqdm \
    jupyter \
    notebook \
    ipywidgets \
    ipykernel \
    jupyterlab

echo "  ✅ All packages installed"

# Register kernel for Jupyter
python -m ipykernel install --user --name=$ENV_NAME --display-name "SpamShield (RTX 5090)"
echo "  ✅ Jupyter kernel registered: 'SpamShield (RTX 5090)'"

# ── Step 7: Verify full stack ─────────────────────────────────────
echo ""
echo "[7/7] Full stack verification..."
python -c "
import torch, transformers, optuna, sklearn, scipy
import numpy as np, pandas as pd

print(f'  torch        : {torch.__version__}')
print(f'  transformers : {transformers.__version__}')
print(f'  optuna       : {optuna.__version__}')
print(f'  scikit-learn : {sklearn.__version__}')
print(f'  scipy        : {scipy.__version__}')
print(f'  numpy        : {np.__version__}')
print(f'  pandas       : {pd.__version__}')

# BF16 matmul benchmark
if torch.cuda.is_available():
    a = torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')
    _ = torch.mm(a, b)
    torch.cuda.synchronize()
    print()
    print(f'  ✅ BF16 matmul OK')
    print(f'  ✅ Compute capability: {torch.cuda.get_device_capability(0)}')
    
    # Check sm_120 (Blackwell)
    cap = torch.cuda.get_device_capability(0)
    if cap[0] >= 12:
        print(f'  ✅ Blackwell (sm_120+) confirmed')
    else:
        print(f'  ⚠️  Not Blackwell: sm_{cap[0]}{cap[1]}')
"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ✅ Setup Complete!                                  ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Activate environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "Start Jupyter:"
echo "  jupyter lab"
echo ""
echo "Select kernel: 'SpamShield (RTX 5090)'"