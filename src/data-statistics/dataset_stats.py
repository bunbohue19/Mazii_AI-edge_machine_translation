#!/usr/bin/env python3
"""
Dataset statistics helper for machine translation

Run:
    python3 src/data-statistics/dataset_stats.py --data data/train/data.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable, List, Mapping, MutableMapping, Sequence


def summarize_numeric(values: Sequence[int]) -> Mapping[str, float]:
    """Return basic stats for a list of integers."""
    if not values:
        return {}
    sorted_vals = sorted(values)

    def pct(p: float) -> int:
        idx = min(int(p * len(sorted_vals)), len(sorted_vals) - 1)
        return sorted_vals[idx]

    return {
        "count": len(values),
        "min": sorted_vals[0],
        "median": median(sorted_vals),
        "mean": mean(sorted_vals),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "max": sorted_vals[-1],
    }


def length_stats(strings: Iterable[str]) -> Mapping[str, Mapping[str, float]]:
    """Compute char/word length summaries for an iterable of strings."""
    chars: List[int] = []
    words: List[int] = []
    for s in strings:
        chars.append(len(s))
        words.append(len(s.split()))
    return {"chars": summarize_numeric(chars), "words": summarize_numeric(words)}


def duplicate_counts(values: Iterable[str]) -> Mapping[str, int]:
    """Count duplicate values and how many rows are repeats beyond the first."""
    counts = Counter(values)
    dup_values = {val: cnt for val, cnt in counts.items() if cnt > 1}
    return {
        "unique_values": len(counts),
        "duplicate_values": len(dup_values),
        "rows_beyond_first": sum(cnt - 1 for cnt in dup_values.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Print dataset statistics.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/train/data.json"),
        help="Path to data.json",
    )
    args = parser.parse_args()

    raw = json.loads(args.data.read_text())
    total_rows = len(raw)

    texts = [row.get("text", "") for row in raw]
    translations = [row.get("translate", "") for row in raw]
    languages = [row.get("targetLanguageCode", "UNKNOWN") for row in raw]

    lang_counts = Counter(languages)
    lang_buckets: MutableMapping[str, List[Mapping[str, str]]] = defaultdict(list)
    for row in raw:
        lang_buckets[row.get("targetLanguageCode", "UNKNOWN")].append(row)

    per_language = {}
    for lang, rows in lang_buckets.items():
        lang_texts = [r.get("text", "") for r in rows]
        lang_trans = [r.get("translate", "") for r in rows]
        per_language[lang] = {
            "rows": len(rows),
            "text_lengths": length_stats(lang_texts),
            "translation_lengths": length_stats(lang_trans),
            "empty_text_rows": sum(1 for s in lang_texts if not s.strip()),
            "empty_translation_rows": sum(1 for s in lang_trans if not s.strip()),
        }

    stats = {
        "rows": total_rows,
        "languages": lang_counts,
        "empty_text_rows": sum(1 for s in texts if not s.strip()),
        "empty_translation_rows": sum(1 for s in translations if not s.strip()),
        "text_length": length_stats(texts),
        "translation_length": length_stats(translations),
        "text_duplicates": duplicate_counts(texts),
        "translation_duplicates": duplicate_counts(translations),
        "pair_duplicates": duplicate_counts(list(zip(texts, translations))),
        "per_language": per_language,
        "sample_text": texts[0] if texts else "",
        "sample_translation": translations[0] if translations else "",
    }

    print(json.dumps(stats, indent=3, ensure_ascii=False))


if __name__ == "__main__":
    main()
