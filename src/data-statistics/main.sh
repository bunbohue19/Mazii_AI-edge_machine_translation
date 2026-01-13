DATASET_NAME="2026-01-13_07-25-54"

python dataset_stats.py \
    --data ../../data/train/$DATASET_NAME.json

# python visualize_dataset.py \
#     --data ../../data/train/$DATASET_NAME.json \
#     --out ../../data/visualization