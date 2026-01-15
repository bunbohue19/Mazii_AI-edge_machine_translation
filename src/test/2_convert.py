
from pathlib import Path
import csv
import json


def convert_jsonl_to_csv(jsonl_path: str, csv_path: str) -> None:
    """Read a JSONL file with one JSON object per line and write it as CSV."""
    jsonl_file = Path(jsonl_path)
    csv_file = Path(csv_path)

    rows = []
    with jsonl_file.open(encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"No data found in {jsonl_file}")

    headers = list(rows[0].keys())
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    with csv_file.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    convert_jsonl_to_csv("data/result_vi_ja.jsonl", "data/result_vi_ja.csv")
