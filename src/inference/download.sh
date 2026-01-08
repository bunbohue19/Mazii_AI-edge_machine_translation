PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"

# echo $PROJECT_ROOT

# HF config
export HF_HOME="$PROJECT_ROOT/model"

hf download Qwen/Qwen3-8B
hf download bunbohue/Mazii-MT-10-06-01_07-01-2026