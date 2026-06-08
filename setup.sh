#!/usr/bin/env bash
# SSeg installer. Creates a conda env, installs the right PyTorch wheel for
# the system's CUDA driver, and installs the rest of requirements.txt.
#
# Usage:
#   bash setup.sh                       # auto-detect, env "spuw"
#   bash setup.sh --env-name myenv      # custom env name
#   bash setup.sh --cuda 11.8           # force CUDA variant
#   bash setup.sh --cpu                 # CPU-only install
#
# After this finishes, run `bash checkpoints/download_ckpts.sh` to grab the
# SAM2 weights, then `python run.py` to verify.

set -euo pipefail

# Ignore the user's ~/.local site-packages. Otherwise `pip install` may
# "satisfy" deps from there instead of the conda env, breaking the CUDA
# variant we explicitly asked for.
export PYTHONNOUSERSITE=1

ENV_NAME="spuw"
CUDA_VARIANT=""        # "", "cu118", "cu121", "cu124", "cpu"
PYTHON_VERSION="3.10"

# ---- arg parsing ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)   ENV_NAME="$2"; shift 2;;
        --cuda)
            case "$2" in
                11.8|11.7|11.6) CUDA_VARIANT="cu118";;
                12.0|12.1|12.2|12.3) CUDA_VARIANT="cu121";;
                12.4|12.5|12.6|12.7|12.8) CUDA_VARIANT="cu124";;
                *) echo "[setup.sh] Unrecognized CUDA $2 — using cu121"; CUDA_VARIANT="cu121";;
            esac
            shift 2
            ;;
        --cpu)        CUDA_VARIANT="cpu"; shift;;
        -h|--help)
            sed -n '2,12p' "$0"; exit 0;;
        *) echo "[setup.sh] Unknown arg $1"; exit 1;;
    esac
done

# ---- detect conda ----
if ! command -v conda &>/dev/null; then
    echo "[setup.sh] conda not found. Install miniconda first." >&2
    exit 1
fi

# Enable `conda activate` inside this script
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ---- detect CUDA if not given ----
if [[ -z "$CUDA_VARIANT" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        # nvidia-smi prints "CUDA Version: 12.6" — grab the major.minor
        CUDA_VER="$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' | head -1)"
        if [[ -n "$CUDA_VER" ]]; then
            echo "[setup.sh] Detected CUDA driver version: $CUDA_VER"
            MAJOR="${CUDA_VER%%.*}"
            MINOR="${CUDA_VER#*.}"
            if [[ "$MAJOR" -ge 12 && "$MINOR" -ge 4 ]] || [[ "$MAJOR" -gt 12 ]]; then
                CUDA_VARIANT="cu124"
            elif [[ "$MAJOR" -ge 12 ]]; then
                CUDA_VARIANT="cu121"
            elif [[ "$MAJOR" -eq 11 && "$MINOR" -ge 8 ]]; then
                CUDA_VARIANT="cu118"
            else
                echo "[setup.sh] CUDA $CUDA_VER is too old for PyTorch 2.x — falling back to CPU."
                CUDA_VARIANT="cpu"
            fi
        else
            echo "[setup.sh] nvidia-smi present but couldn't parse CUDA — falling back to CPU."
            CUDA_VARIANT="cpu"
        fi
    else
        echo "[setup.sh] No nvidia-smi found — installing CPU-only PyTorch."
        CUDA_VARIANT="cpu"
    fi
fi
echo "[setup.sh] PyTorch variant: $CUDA_VARIANT"

# ---- create env ----
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup.sh] Conda env '$ENV_NAME' already exists — reusing."
else
    echo "[setup.sh] Creating conda env '$ENV_NAME' (python $PYTHON_VERSION)..."
    conda create -n "$ENV_NAME" "python=${PYTHON_VERSION}" -y
fi

conda activate "$ENV_NAME"
echo "[setup.sh] Active env: $(python -c 'import sys; print(sys.prefix)')"

# Make sure pip is current — older pip has trouble with the torch index.
# --no-user pins installs to the env (defense in depth on top of PYTHONNOUSERSITE).
python -m pip install --no-user --upgrade pip wheel

# ---- install PyTorch ----
if [[ "$CUDA_VARIANT" == "cpu" ]]; then
    PYTORCH_INDEX="https://download.pytorch.org/whl/cpu"
else
    PYTORCH_INDEX="https://download.pytorch.org/whl/${CUDA_VARIANT}"
fi
echo "[setup.sh] Installing torch+torchvision from $PYTORCH_INDEX ..."
pip install --no-user --force-reinstall torch torchvision --index-url "$PYTORCH_INDEX"

# Verify torch can see GPU (warn but don't fail if not)
python - <<'PYCHECK'
import torch
print(f"[setup.sh] torch {torch.__version__}  cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"           device: {torch.cuda.get_device_name(0)}")
PYCHECK

# ---- install rest ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REQ="${SCRIPT_DIR}/requirements.txt"
if [[ ! -f "$REQ" ]]; then
    echo "[setup.sh] requirements.txt not found at $REQ" >&2
    exit 1
fi
echo "[setup.sh] Installing remaining deps from requirements.txt ..."
pip install --no-user -r "$REQ"

# ---- smoke import ----
echo "[setup.sh] Smoke importing the things SSeg/run.py uses..."
python - <<'PYCHECK'
import importlib, sys
mods = [
    'torch', 'torchvision', 'cv2', 'numpy', 'pandas', 'matplotlib',
    'sklearn', 'skimage', 'tqdm', 'hydra', 'iopath', 'torchmetrics',
    'PIL', 'scipy',
]
fail = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, '__version__', '?')
        print(f"  ok  {m:14s} {v}")
    except Exception as e:
        print(f"  FAIL {m:14s} {e}")
        fail.append(m)
if fail:
    print(f"\n[setup.sh] {len(fail)} module(s) failed to import: {fail}")
    sys.exit(1)
print("\n[setup.sh] All deps importable.")
PYCHECK

# ---- done ----
cat <<EOF

================================================================
Install OK.

Next steps:
  conda activate $ENV_NAME
  bash checkpoints/download_ckpts.sh   # ~900 MB SAM2 checkpoint
  python run.py                        # runs the demo experiment

================================================================
EOF
