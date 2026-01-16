PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
echo $PROJECT_ROOT

# HF config
export HF_HOME="$PROJECT_ROOT/model"
export CUDA_VISIBLE_DEVICES=0
export VERSION="v1.10"

MODEL_PATH="$HF_HOME/hub/models--tencent--HY-MT1.5-1.8B/snapshots/dbad03788f49709801014c95d481a514c272ca52"

python main.py \
    --model-path $MODEL_PATH 