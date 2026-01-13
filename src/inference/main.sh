PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"

# echo $PROJECT_ROOT

# HF config
export HF_HOME="$PROJECT_ROOT/model"
export CUDA_VISIBLE_DEVICES=0
export VERSION="v1.6"

python main.py \
    --model-path "$HF_HOME/" \
    --adapter-path "$HF_HOME/"