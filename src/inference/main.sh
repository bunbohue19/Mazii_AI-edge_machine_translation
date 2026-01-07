PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"

# echo $PROJECT_ROOT

# HF config
export HF_HOME="$PROJECT_ROOT/model"
export CUDA_VISIBLE_DEVICES=0
export VERSION="v1.5"

python main.py \
    --model-path "$HF_HOME/hub/models--unsloth--Qwen2.5-7B-Instruct/snapshots/a75c9dc945567a9b6f568b8503a0307731607bee" \
    --adapter-path "$HF_HOME/hub/models--bunbohue--Mazii-MT-13-28-30_06-01-2026/snapshots/1fd6c79ae4932f74876490322622b1e20321510b"