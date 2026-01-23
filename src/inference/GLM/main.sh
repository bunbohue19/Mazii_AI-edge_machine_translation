PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"

# HF config
export HF_HOME="$PROJECT_ROOT/model"
export CUDA_VISIBLE_DEVICES=0
# export CUDA_HOME=/usr/local/cuda
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export VERSION="v1.9"

MODEL_PATH="$HF_HOME/hub/models--zai-org--GLM-4-9B-0414/snapshots/645b8482494e31b6b752272bf7f7f273ef0f3caf"
# MODEL_PATH="$HF_HOME/hub/models--bunbohue--GLM-4-9B-0414-FP8/snapshots/d4e39d5b68b0e519e75e587b6b29209cf2ca036f"

python main.py \
    --model-path $MODEL_PATH 