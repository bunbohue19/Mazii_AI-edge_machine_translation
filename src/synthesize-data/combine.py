import os
import json
from pathlib import Path

BASE_DIR = Path("/home/locseo/Mazii_AI-edge_machine_translation")
INPUT_DIR = BASE_DIR / "data" / "synthetic-data" / "output"
TRAIN_DIR = BASE_DIR / "data" / "train"

NEW_DATA_FILE = TRAIN_DIR / f"{os.getenv("NEW_DATA_NAME")}.json"
BASE_DATA_FILE = TRAIN_DIR / f"{os.getenv("BASE_DATA_NAME")}.json"
MERGED_OUTPUT_FILE = TRAIN_DIR / f"{os.getenv("MERGED_OUTPUT_NAME")}.json"


def load_list(path: Path) -> list:
    """Load JSON content if it's a list; otherwise return an empty list."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def write_json(path: Path, payload: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def merge_synthetic_outputs() -> None:
    """Merge all synthetic *.json files into new_data.json."""
    result = []
    for file_path in sorted(INPUT_DIR.glob("*.json")):
        result.extend(load_list(file_path))
    write_json(NEW_DATA_FILE, result)


def merge_train_files() -> None:
    """Merge base data.json and new_data.json into combined_data.json."""
    combined = []
    if BASE_DATA_FILE.exists():
        combined.extend(load_list(BASE_DATA_FILE))
    if NEW_DATA_FILE.exists():
        combined.extend(load_list(NEW_DATA_FILE))
    write_json(MERGED_OUTPUT_FILE, combined)


if __name__ == "__main__":
    merge_synthetic_outputs()
    merge_train_files()
