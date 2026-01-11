PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

DATASET_NAME="[JA-JA]_実用日本語表現辞典"

TEXT_COLUMN="text"
DATASET_PATH="$PROJECT_ROOT/data/synthetic-data/input/$DATASET_NAME.jsonl"
MODEL_NAME="gemini-3-flash-preview"
TARGET_LANG_CODE="Vietnamese"
START_ID=10000
END_ID=20000
OUTPUT_PATH="$PROJECT_ROOT/data/synthetic-data/output/$TIMESTAMP-$TARGET_LANG_CODE.json"

python translation_runner.py \
  --text_column "$TEXT_COLUMN" \
  --dataset_path "$DATASET_PATH" \
  --model_name "$MODEL_NAME" \
  --target_lang_code "$TARGET_LANG_CODE" \
  --start_id "$START_ID" \
  --end_id "$END_ID" \
  --output_path "$OUTPUT_PATH"