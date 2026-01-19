PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"

# HF config
export HF_HOME="$PROJECT_ROOT/model"
export CUDA_VISIBLE_DEVICES=0
export VERSION="v1.9"

MODEL_PATH="$HF_HOME/hub/models--zai-org--GLM-4-9B-0414/snapshots/645b8482494e31b6b752272bf7f7f273ef0f3caf"
# MODEL_PATH="$HF_HOME/hub/models--unsloth--GLM-4-9B-0414-bnb-4bit/snapshots/fed7dd6eb06bb17e9b8850459a530ac6242fac45"

python main.py \
    --model-path $MODEL_PATH 