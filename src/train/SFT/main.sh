export CUDA_VISIBLE_DEVICES=0
export HF_HOME=../../../model
export MODEL_ID="tencent/HY-MT1.5-1.8B"
export DATASET_NAME="2026-01-13_07-25-54"

python main.py

# python unsloth_main.py