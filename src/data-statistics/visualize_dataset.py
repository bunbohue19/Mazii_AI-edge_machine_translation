#!/usr/bin/env python3
"""
Generate quick visualizations for data/train/data.json.

Outputs:
    - language_counts.png : bar chart of targetLanguageCode counts
    - source_words_hist.png / target_words_hist.png : word-count histograms (capped at p99 to tame tails)

Usage:
    python3 src/data-statistics/visualize_dataset.py --data data/train/data.json --out data/visualization
"""
from __future__ import annotations

import argparse
import json
import matplotlib.pyplot as plt # type: ignore
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence


def percentile(values: Sequence[int], p: float) -> int:
    """Return percentile value (simple lower rank)."""
    if not values:
        return 0
    sorted_vals = sorted(values)
    idx = min(int(p * len(sorted_vals)), len(sorted_vals) - 1)
    return sorted_vals[idx]


def word_lengths(texts: Iterable[str]) -> List[int]:
    return [len(t.split()) for t in texts]


def plot_language_counts(counts: Mapping[str, int], outfile: Path) -> None:
    langs = list(counts.keys())
    vals = [counts[k] for k in langs]
    plt.figure(figsize=(6, 4))
    plt.bar(langs, vals, color="#4a90e2")
    plt.title("Language distribution")
    plt.ylabel("Rows")
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_histogram(
    values: Sequence[int],
    title: str,
    xlabel: str,
    outfile: Path,
    bins: int = 50,
    cap_percentile: float = 0.99,
) -> None:
    """Plot histogram with optional cap at percentile to reduce long-tail distortion."""
    if not values:
        return
    cap = percentile(values, cap_percentile)
    capped = [min(v, cap) for v in values]
    plt.figure(figsize=(8, 4))
    plt.hist(capped, bins=bins, color="#f5a623", edgecolor="black", alpha=0.8)
    plt.title(f"{title} (values capped at p{int(cap_percentile*100)})")
    plt.xlabel(xlabel)
    plt.ylabel("Rows")
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize dataset stats.")
    parser.add_argument("--data", type=Path, default=Path("data/train/data.json"))
    parser.add_argument("--out", type=Path, default=Path("data/visualization"), help="Output directory")
    args = parser.parse_args()

    rows = json.loads(args.data.read_text())
    texts = [r.get("text", "") for r in rows]
    translations = [r.get("translate", "") for r in rows]
    languages = [r.get("targetLanguageCode", "UNKNOWN") for r in rows]

    lang_counts = Counter(languages)
    plot_language_counts(lang_counts, args.out / "language_counts.png")

    plot_histogram(
        word_lengths(texts),
        title="Source word counts",
        xlabel="Words in source text",
        outfile=args.out / "source_words_hist.png",
    )
    plot_histogram(
        word_lengths(translations),
        title="Target word counts",
        xlabel="Words in translation",
        outfile=args.out / "target_words_hist.png",
    )


if __name__ == "__main__":
    main()
