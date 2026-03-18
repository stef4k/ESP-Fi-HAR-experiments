#!/bin/bash
#SBATCH --job-name=espfi-table4-rf-fast-transformer
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

# --- user-configurable ---
# Preferred: full conda env path on scratch (quota-safe).
ENV_PATH="/gpfs/scratch/${USER}/conda-envs/espfi"
# Fallback: named env (set this only if you really use named envs).
ENV_NAME="har"
DATA_ROOT="${SLURM_SUBMIT_DIR}/ModelCode/Data/rf-based_data"
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/ModelCode/benchmark_outputs_rf_fast_transformer"

mkdir -p "${SLURM_SUBMIT_DIR}/logs"

module purge
module load anaconda3/2023.09-0/none-none

if [[ -n "${ENV_PATH}" && -d "${ENV_PATH}" ]]; then
  source activate "${ENV_PATH}"
elif [[ -n "${ENV_NAME}" ]] && conda info --envs | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  source activate "${ENV_NAME}"
else
  echo "Conda environment not found."
  echo "Tried ENV_PATH=${ENV_PATH} and ENV_NAME=${ENV_NAME}"
  conda info --envs
  exit 1
fi

cd "${SLURM_SUBMIT_DIR}"

# Preflight: fail fast if Torch does not support the allocated GPU arch.
python - <<'PY'
import sys
import torch

print("Torch version:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)

if not torch.cuda.is_available():
    print("CUDA available: False (deep models will run on CPU).")
    raise SystemExit(0)

device_name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
required_arch = f"sm_{cap[0]}{cap[1]}"
arch_list = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []

print("GPU:", device_name)
print("Required arch:", required_arch)
print("Torch arch list:", " ".join(arch_list) if arch_list else "(unknown)")

if arch_list and required_arch not in arch_list:
    print("\nERROR: Incompatible PyTorch CUDA build for this GPU.")
    print("Fix your env (once), then resubmit:")
    print(
        "python -m pip install --upgrade --index-url "
        "https://download.pytorch.org/whl/cu118 torch torchvision torchaudio"
    )
    raise SystemExit(2)
PY

python ModelCode/table4_benchmark.py \
  --dataset-kind rf_csv \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --scenes 1,2,3 \
  --ml-models SVM,RF \
  --deep-models LSTM,GRU,CNN,Transformer \
  --no-compare-paper \
  --num-workers 4 \
  --ml-n-jobs 4
