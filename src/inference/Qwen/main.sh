PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"

# echo $PROJECT_ROOT

# HF config
export HF_HOME="$PROJECT_ROOT/model"
export CUDA_VISIBLE_DEVICES=0
export VERSION="v1.5"

MODEL_PATH="$HF_HOME/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
ADAPTER_PATH="$HF_HOME/hub/models--bunbohue--Mazii-MT-09-38-43_13-01-2026/snapshots/14c241687202ef6b3326538b3c89bd27b97759a0"

python main.py \
    --model-path $MODEL_PATH \
    --adapter-path $ADAPTER_PATH