#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
ENV_NAME="${ENV_NAME:-v13-brepgen}"
ENV_PREFIX="${ENV_PREFIX:-}"
PYTHON_BIN="${PYTHON:-}"

V13_BOOTSTRAP_MODE="${V13_BOOTSTRAP_MODE:-vqvae}"  # vqvae | full
V13_REQUIRE_CUDA="${V13_REQUIRE_CUDA:-0}"
V13_FORCE_CU128="${V13_FORCE_CU128:-0}"
V13_INSTALL_TORCH="${V13_INSTALL_TORCH:-auto}"     # auto | 0 | 1
V13_SKIP_INSTALL="${V13_SKIP_INSTALL:-0}"
V13_TORCH_INDEX_URL="${V13_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
V13_PYPI_INDEX_URL="${V13_PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

cd "$REPO_ROOT"

create_or_activate_env() {
  if [[ -n "$PYTHON_BIN" ]]; then
    echo "Using explicit Python: $PYTHON_BIN"
    return
  fi

  if command -v micromamba >/dev/null 2>&1; then
    if [[ -n "$ENV_PREFIX" ]]; then
      if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
        micromamba create -y -p "$ENV_PREFIX" -f environment.server.yml
      fi
      eval "$(micromamba shell hook --shell bash)"
      micromamba activate "$ENV_PREFIX"
    else
      if ! micromamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        micromamba create -y -n "$ENV_NAME" -f environment.server.yml
      fi
      eval "$(micromamba shell hook --shell bash)"
      micromamba activate "$ENV_NAME"
    fi
    PYTHON_BIN="python"
    return
  fi

  if command -v mamba >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    if [[ -n "$ENV_PREFIX" ]]; then
      if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
        mamba env create -p "$ENV_PREFIX" -f environment.server.yml
      fi
      conda activate "$ENV_PREFIX"
    else
      if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        mamba env create -f environment.server.yml
      fi
      conda activate "$ENV_NAME"
    fi
    PYTHON_BIN="python"
    return
  fi

  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    if [[ -n "$ENV_PREFIX" ]]; then
      if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
        conda env create -p "$ENV_PREFIX" -f environment.server.yml
      fi
      conda activate "$ENV_PREFIX"
    else
      if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        conda env create -f environment.server.yml
      fi
      conda activate "$ENV_NAME"
    fi
    PYTHON_BIN="python"
    return
  fi

  PYTHON_BIN="python"
  echo "No conda/mamba found; using existing Python: $PYTHON_BIN"
}

python_has_module() {
  "$PYTHON_BIN" - "$1" <<'PY'
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec(sys.argv[1]) is not None else 1)
PY
}

install_torch_cu128() {
  echo "Installing PyTorch from $V13_TORCH_INDEX_URL"
  "$PYTHON_BIN" -m pip install --force-reinstall torch torchvision torchaudio \
    --index-url "$V13_TORCH_INDEX_URL"
}

uninstall_known_torch_cuda_wheels() {
  "$PYTHON_BIN" -m pip uninstall -y \
    torch torchvision torchaudio triton \
    cuda-bindings cuda-pathfinder cuda-toolkit \
    nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
    nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand nvidia-cusolver \
    nvidia-cusparse nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink \
    nvidia-nvshmem-cu13 nvidia-nvtx >/dev/null 2>&1 || true
}

create_or_activate_env

if [[ "$V13_SKIP_INSTALL" == "1" ]]; then
  echo "V13_SKIP_INSTALL=1: checking the current environment without installing packages."
else
  "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

  if [[ "$V13_FORCE_CU128" == "1" ]]; then
    echo "V13_FORCE_CU128=1: removing known torch/CUDA pip wheels first."
    uninstall_known_torch_cuda_wheels
    install_torch_cu128
  elif [[ "$V13_INSTALL_TORCH" == "1" ]]; then
    install_torch_cu128
  elif [[ "$V13_INSTALL_TORCH" == "auto" ]]; then
    if ! python_has_module torch; then
      install_torch_cu128
    fi
  fi

  "$PYTHON_BIN" -m pip install --upgrade \
    "numpy<2.3" \
    "diffusers==0.35.1" \
    "transformers==4.57.3" \
    "huggingface_hub<1.0" \
    einops accelerate safetensors tqdm zstandard scipy \
    -i "$V13_PYPI_INDEX_URL"
fi

"$PYTHON_BIN" - <<'PY'
import importlib.util
import os
import sys

mode = os.environ.get("V13_BOOTSTRAP_MODE", "vqvae")
required = [
    "torch",
    "numpy",
    "transformers",
    "diffusers",
    "einops",
    "zstandard",
]
if mode == "full":
    required.append("OCC.Core.TopoDS")

print("python:", sys.executable)
print("prefix:", sys.prefix)
print("bootstrap mode:", mode)

missing = []
for name in required:
    try:
        spec = importlib.util.find_spec(name)
    except Exception as exc:
        spec = None
        print(f"{name}: missing ({exc})")
    else:
        print(f"{name}: {'ok' if spec is not None else 'missing'}")
    if spec is None:
        missing.append(name)

if missing:
    print("Missing modules: " + ", ".join(missing), file=sys.stderr)
    if "OCC.Core.TopoDS" in missing:
        print(
            "OCC is only required for STEP reconstruction/validity checks. "
            "For VQ-VAE training, keep V13_BOOTSTRAP_MODE=vqvae.",
            file=sys.stderr,
        )
    sys.exit(2)

from diffusers import VQModel
import torch

print("diffusers.VQModel:", VQModel)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

require_cuda = os.environ.get("V13_REQUIRE_CUDA", "0") == "1"
if not torch.cuda.is_available():
    if require_cuda:
        print("CUDA is required but torch.cuda.is_available() is false.", file=sys.stderr)
        sys.exit(3)
    print("CUDA is not visible; package import checks passed.")
    sys.exit(0)

print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("arch list:", torch.cuda.get_arch_list())

try:
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("cuda matmul ok:", float(y.mean().detach().cpu()))
except Exception as exc:
    print("CUDA kernel smoke test failed:", repr(exc), file=sys.stderr)
    print(
        "If this is an RTX 5090 / Blackwell host, rerun with "
        "V13_FORCE_CU128=1 to install the official PyTorch cu128 wheels.",
        file=sys.stderr,
    )
    sys.exit(4)
PY

echo "Environment bootstrap complete."
echo "VQ-VAE mode does not require OCC. Use V13_BOOTSTRAP_MODE=full only before reconstruction gates."
