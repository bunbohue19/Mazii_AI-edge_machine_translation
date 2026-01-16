PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"

# echo $PROJECT_ROOT

# HF config
export HF_HOME="$PROJECT_ROOT/model"

# Base model
# hf download Qwen/Qwen3-8B
# hf download Qwen/Qwen3-4B
# hf download Aratako/Qwen3-8B-NSFW-JP
hf download google/translategemma-4b-it

# Adapter
# hf download bunbohue/Mazii-MT-09-38-43_13-01-2026
# hf download bunbohue/Mazii-MT-10-25-26_13-01-2026
# hf download bunbohue/Mazii-MT-07-10-54_14-01-2026