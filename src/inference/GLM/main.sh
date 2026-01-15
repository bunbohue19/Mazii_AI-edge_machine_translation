PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
echo $PROJECT_ROOT

# HF config
export HF_HOME="$PROJECT_ROOT/model"
export CUDA_VISIBLE_DEVICES=0
export VERSION="v1.9"

MODEL_PATH="$HF_HOME/"
ADAPTER_PATH="$HF_HOME/"

python main.py \
    --model-path $MODEL_PATH \
    --adapter-path $ADAPTER_PATH