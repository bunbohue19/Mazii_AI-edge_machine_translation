PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"

# echo $PROJECT_ROOT

# HF config
export HF_HOME="$PROJECT_ROOT/model"
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="$HF_HOME/hub/models--google--translategemma-4b-it/snapshots/4eded5e6860cc44d717912be26cc02d37606cf4f"

python main.py \
    --model-path $MODEL_PATH 