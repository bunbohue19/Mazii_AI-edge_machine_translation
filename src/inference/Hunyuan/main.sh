PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
echo $PROJECT_ROOT

# HF config
export HF_HOME="$PROJECT_ROOT/model"
export CUDA_VISIBLE_DEVICES=0
export VERSION="v1.10"
export FLASHINFER_DISABLE_VERSION_CHECK=1

MODEL_PATH="$HF_HOME/hub/models--tencent--HY-MT1.5-1.8B/snapshots/dbad03788f49709801014c95d481a514c272ca52"
ADAPTER_PATH="$PROJECT_ROOT/output/eUp-MT-08-01-55_25-03-2026/checkpoint-1000"

python main.py \
    --model-path $MODEL_PATH \
    --adapter-path $ADAPTER_PATH