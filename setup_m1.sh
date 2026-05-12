#!/usr/bin/env bash
# =============================================================================
# setup_m1.sh  —  One-time environment setup for Apple M1 (arm64)
# Run with:  bash setup_m1.sh
# Assumes:   conda is already installed (e.g. via miniforge/mambaforge)
#            Homebrew is installed
# =============================================================================
set -e

# ── 1. LLVM 14 via Homebrew ──────────────────────────────────────────────────
# echo ">>> Installing LLVM 14..."
# brew install llvm@14

# LLVM_PREFIX=$(brew --prefix llvm@14)
# LLVM_LINE="export PATH=\"${LLVM_PREFIX}/bin:\$PATH\""
# LDFLAGS_LINE="export LDFLAGS=\"-L${LLVM_PREFIX}/lib \$LDFLAGS\""
# CPPFLAGS_LINE="export CPPFLAGS=\"-I${LLVM_PREFIX}/include \$CPPFLAGS\""

# # Append to ~/.zshrc only if not already present
# grep -qxF "$LLVM_LINE" ~/.zshrc || echo "$LLVM_LINE"    >> ~/.zshrc
# grep -qxF "$LDFLAGS_LINE" ~/.zshrc || echo "$LDFLAGS_LINE" >> ~/.zshrc
# grep -qxF "$CPPFLAGS_LINE" ~/.zshrc || echo "$CPPFLAGS_LINE" >> ~/.zshrc

# export PATH="${LLVM_PREFIX}/bin:$PATH"
# echo "   clang: $(which clang)  version: $(clang --version | head -1)"
# echo "   opt:   $(which opt)"

# ── 2. Conda environment ──────────────────────────────────────────────────────
echo ">>> Creating conda env 'compiler_opt' (Python 3.10)..."
conda create -n compiler_opt python=3.10 -y

# Activate inside script (works in bash with conda init already done)
eval "$(conda shell.bash hook)"
conda activate compiler_opt

# ── 3. PyTorch  (MPS backend ships in stock conda pytorch) ───────────────────
echo ">>> Installing PyTorch..."
conda install pytorch torchvision torchaudio -c pytorch -y

TORCH_VER=$(python -c "import torch; v=torch.__version__; print(v.split('+')[0])")
echo "   torch $TORCH_VER  MPS=$(python -c 'import torch; print(torch.backends.mps.is_available())')"

# ── 4. PyTorch Geometric  (CPU wheels — MPS support in PyG is partial) ───────
# NOTE: We deliberately use +cpu wheels here.  The GNN forward pass runs on
# CPU even on M1; MPS is reserved for the PPO actor/critic dense layers.
echo ">>> Installing PyTorch Geometric (CPU wheels)..."
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f "https://data.pyg.org/whl/torch-${TORCH_VER}+cpu.html"

python -c "import torch_geometric; print('   PyG', torch_geometric.__version__, '✓')"

# ── 5. Remaining dependencies ─────────────────────────────────────────────────
echo ">>> Installing remaining dependencies..."
pip install \
    "tree-sitter==0.20.4" \
    "tree-sitter-languages>=1.8.0" \
    networkx \
    scikit-learn \
    matplotlib seaborn \
    pandas numpy \
    tqdm \
    pyyaml \
    kaggle \
    jupyterlab ipykernel

# Register kernel so Jupyter can find the env
python -m ipykernel install --user --name compiler_opt --display-name "Python (compiler_opt)"

# ── 6. Quick smoke test ───────────────────────────────────────────────────────
echo ""
echo ">>> Smoke test..."
python - <<'EOF'
import torch
import torch_geometric
from torch_geometric.data import HeteroData
import tree_sitter_languages
print(f"  torch        {torch.__version__}")
print(f"  mps avail    {torch.backends.mps.is_available()}")
print(f"  torch_geo    {torch_geometric.__version__}")
print(f"  tree-sitter  ok")
EOF

echo ""
echo "=== Setup complete ==="
echo "To activate: conda activate compiler_opt"
echo "To start:    jupyter lab"
