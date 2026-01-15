PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"

# echo $PROJECT_ROOT

# HF config
export HF_HOME="$PROJECT_ROOT/model"
export CUDA_VISIBLE_DEVICES=0
export VERSION="v1.7"

MODEL_PATH="$HF_HOME/hub/models--Aratako--Qwen3-8B-NSFW-JP/snapshots/610294652e72d3a0a2873e3f481c96dbc1890015"
ADAPTER_PATH="$HF_HOME/hub/models--bunbohue--Mazii-MT-07-10-54_14-01-2026/snapshots/b98bc3db3225fd9e48389ad4f29f7101d68fee6c"

python main.py \
    --model-path $MODEL_PATH \
    --adapter-path $ADAPTER_PATH