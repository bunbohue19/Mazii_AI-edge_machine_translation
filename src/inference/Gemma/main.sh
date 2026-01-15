PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"

# echo $PROJECT_ROOT

# HF config
export HF_HOME="$PROJECT_ROOT/model"
export CUDA_VISIBLE_DEVICES=0
export VERSION="v1.8"

MODEL_PATH="$HF_HOME/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767"
# ADAPTER_PATH="$HF_HOME/"

python main.py \
    --model-path $MODEL_PATH \
    # --adapter-path $ADAPTER_PATH