PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"

# echo $PROJECT_ROOT

# HF config
export HF_HOME="$PROJECT_ROOT/model"
export CUDA_VISIBLE_DEVICES=0
export VERSION="v1.5"

python main.py \
    --model-path "$HF_HOME/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218" \
    --adapter-path "$HF_HOME/hub/models--bunbohue--Mazii-MT-10-06-01_07-01-2026/snapshots/556df31f7c87d2c98ba5983d2eb914b168de44c8"