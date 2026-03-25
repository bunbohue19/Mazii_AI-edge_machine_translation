"""
Benchmark script for Hunyuan MT model + LoRA adapter.

Evaluates translation quality using:
  - Dataset : FLORES-200 via openlanguagedata/flores_plus
  - Metric  : wmt22-comet-da (Unbabel/wmt22-comet-da)

Usage:
    python benchmark.py \
        --src-lang Japanese \
        --tgt-lang Vietnamese \
        --n-samples 200 \
        --server-url http://localhost:8501 \
        --output-dir ./results
"""

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
from datasets import load_dataset
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Language mappings
# --------------------------------------------------------------------------- #

# Maps server target_lang_code values → openlanguagedata/flores_plus codes
# (iso_639_3 + "_" + iso_15924 as stored in the dataset)
LANG_TO_FLORES: Dict[str, str] = {
    "Japanese":            "jpn_Jpan",
    "Vietnamese":          "vie_Latn",
    "Traditional Chinese": "cmn_Hant",  # flores_plus uses cmn, not zho
    "Indonesian":          "ind_Latn",
    "English":             "eng_Latn",
}


# --------------------------------------------------------------------------- #
# Translation client
# --------------------------------------------------------------------------- #

async def translate_one(
    session: aiohttp.ClientSession,
    server_url: str,
    text: str,
    target_lang_code: str,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    retries: int = 3,
) -> Tuple[Optional[str], Optional[Dict]]:
    """Send one translation request; returns (translated_text, usage) or (None, None) on failure."""
    payload = {
        "text": text,
        "target_lang_code": target_lang_code,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    async with semaphore:
        for attempt in range(1, retries + 1):
            try:
                async with session.post(
                    f"{server_url}/v1/translate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return data.get("translated_text"), data.get("usage")
            except Exception as exc:
                logger.warning(f"Attempt {attempt}/{retries} failed: {exc}")
                if attempt < retries:
                    await asyncio.sleep(2 ** attempt)
    return None, None


async def translate_batch(
    sources: List[str],
    server_url: str,
    target_lang_code: str,
    temperature: float,
    max_tokens: int,
    concurrency: int,
) -> Tuple[List[Optional[str]], List[Optional[Dict]]]:
    """Translate a list of source sentences concurrently."""
    semaphore = asyncio.Semaphore(concurrency)
    translations: List[Optional[str]] = [None] * len(sources)
    usages: List[Optional[Dict]] = [None] * len(sources)

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            translate_one(session, server_url, text, target_lang_code, temperature, max_tokens, semaphore)
            for text in sources
        ]
        results = await async_tqdm.gather(*tasks, desc="Translating")

    for i, (mt, usage) in enumerate(results):
        translations[i] = mt
        usages[i] = usage

    return translations, usages


# --------------------------------------------------------------------------- #
# FLORES-200 loader
# --------------------------------------------------------------------------- #

def _flores_code_to_filter(flores_code: str) -> Tuple[str, str]:
    """Split 'jpn_Jpan' → ('jpn', 'Jpan') for row-level filtering."""
    iso3, script = flores_code.split("_", 1)
    return iso3, script


def load_flores(
    src_lang: str,
    tgt_lang: str,
    n_samples: Optional[int],
    ds,
) -> Tuple[List[str], List[str]]:
    """Extract a parallel sentence pair from a pre-loaded flores_plus dataset.

    ds must already be loaded (load once in main, reuse across pairs).
    """
    src_iso3, src_script = _flores_code_to_filter(LANG_TO_FLORES[src_lang])
    tgt_iso3, tgt_script = _flores_code_to_filter(LANG_TO_FLORES[tgt_lang])

    logger.info(f"Extracting [{LANG_TO_FLORES[src_lang]} → {LANG_TO_FLORES[tgt_lang]}]")

    src_rows = {
        row["id"]: row["text"]
        for row in ds
        if row["iso_639_3"] == src_iso3 and row["iso_15924"] == src_script
    }
    tgt_rows = {
        row["id"]: row["text"]
        for row in ds
        if row["iso_639_3"] == tgt_iso3 and row["iso_15924"] == tgt_script
    }

    common_ids = sorted(src_rows.keys() & tgt_rows.keys())
    if not common_ids:
        available = sorted({
            f"{r['iso_639_3']}_{r['iso_15924']}"
            for r in ds
            if r["iso_639_3"] in (src_iso3, tgt_iso3)
        })
        raise ValueError(
            f"No aligned sentences found for "
            f"{LANG_TO_FLORES[src_lang]} / {LANG_TO_FLORES[tgt_lang]}. "
            f"Available codes for those iso3s: {available}"
        )

    sources    = [src_rows[i] for i in common_ids]
    references = [tgt_rows[i] for i in common_ids]

    if n_samples and n_samples < len(sources):
        sources    = sources[:n_samples]
        references = references[:n_samples]

    logger.info(f"Loaded {len(sources)} sentence pairs")
    return sources, references


# --------------------------------------------------------------------------- #
# wmt22-comet-da scorer
# --------------------------------------------------------------------------- #

def score_comet(
    sources: List[str],
    hypotheses: List[str],
    references: List[str],
    batch_size: int,
    gpus: int,
) -> Tuple[List[float], float]:
    """Score translations with wmt22-comet-da; returns (segment_scores, system_score)."""
    from comet import download_model, load_from_checkpoint

    logger.info("Loading wmt22-comet-da model (downloaded on first run)…")
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)

    data = [
        {"src": src, "mt": mt, "ref": ref}
        for src, mt, ref in zip(sources, hypotheses, references)
    ]

    logger.info(f"Scoring {len(data)} segments with wmt22-comet-da (batch_size={batch_size}, gpus={gpus})…")
    output = model.predict(data, batch_size=batch_size, gpus=gpus)

    segment_scores: List[float] = output.scores
    system_score: float = output.system_score
    return segment_scores, system_score


# --------------------------------------------------------------------------- #
# Results serialization
# --------------------------------------------------------------------------- #

def save_results(
    output_dir: Path,
    src_lang: str,
    tgt_lang: str,
    sources: List[str],
    hypotheses: List[Optional[str]],
    references: List[str],
    segment_scores: List[float],
    system_score: float,
    usages: List[Optional[Dict]],
    elapsed_s: float,
    args_dict: Dict,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"benchmark_{src_lang}_to_{tgt_lang}_{timestamp}.json".replace(" ", "_")

    total_tokens = sum((u or {}).get("total_tokens", 0) for u in usages)
    failed = sum(1 for h in hypotheses if h is None)

    payload = {
        "meta": {
            "timestamp": timestamp,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "n_samples": len(sources),
            "n_failed": failed,
            "elapsed_seconds": round(elapsed_s, 2),
            "avg_latency_ms": round(elapsed_s / len(sources) * 1000, 1),
            "total_tokens": total_tokens,
            "args": args_dict,
        },
        "scores": {
            "comet_da_system": round(system_score, 6),
            "comet_da_segment_mean": round(sum(segment_scores) / len(segment_scores), 6),
            "comet_da_segment_min":  round(min(segment_scores), 6),
            "comet_da_segment_max":  round(max(segment_scores), 6),
        },
        "samples": [
            {
                "id": i,
                "source":     src,
                "hypothesis": hyp,
                "reference":  ref,
                "comet_da_score": round(seg, 6),
                "usage": usage,
            }
            for i, (src, hyp, ref, seg, usage) in enumerate(
                zip(sources, hypotheses, references, segment_scores, usages)
            )
        ],
    }

    result_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    logger.info(f"Results saved → {result_file}")
    return result_file


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark Hunyuan MT with FLORES-200 + wmt22-comet-da")
    p.add_argument("--server-url",   default="http://localhost:8501",
                   help="Base URL of the running SGLang translation server")
    p.add_argument("--src-lang",     default="Japanese",
                   choices=list(LANG_TO_FLORES), help="Source language")
    p.add_argument("--tgt-lang",     default="Vietnamese",
                   choices=list(LANG_TO_FLORES), help="Target language")
    p.add_argument("--flores-split", default="devtest", choices=["dev", "devtest"],
                   help="FLORES-200 split to use")
    p.add_argument("--n-samples",    type=int, default=None,
                   help="Max sentences to evaluate (default: full split)")
    p.add_argument("--temperature",  type=float, default=0.20,
                   help="Sampling temperature (match your serving config)")
    p.add_argument("--max-tokens",   type=int, default=1024)
    p.add_argument("--concurrency",  type=int, default=8,
                   help="Number of concurrent translation requests")
    p.add_argument("--comet-batch",  type=int, default=8,
                   help="Batch size for wmt22-comet-da inference")
    p.add_argument("--gpus",         type=int, default=1,
                   help="GPUs to use for XCOMET-XXL scoring")
    p.add_argument("--output-dir",   type=Path, default=Path("results"),
                   help="Directory to write result JSON files")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load flores_plus dataset once — reused across all language pairs
    logger.info(f"Loading openlanguagedata/flores_plus split={args.flores_split}…")
    flores_ds = load_dataset("openlanguagedata/flores_plus", split=args.flores_split)

    # 2. Extract parallel sentences
    sources, references = load_flores(
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        n_samples=args.n_samples,
        ds=flores_ds,
    )

    # 3. Translate
    logger.info(f"Translating {len(sources)} sentences → {args.tgt_lang} via {args.server_url}")
    t0 = time.monotonic()
    hypotheses, usages = asyncio.run(
        translate_batch(
            sources=sources,
            server_url=args.server_url,
            target_lang_code=args.tgt_lang,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
        )
    )
    elapsed = time.monotonic() - t0

    failed = sum(1 for h in hypotheses if h is None)
    logger.info(f"Translation done in {elapsed:.1f}s — {failed}/{len(sources)} failed")

    # Replace failed translations with empty string so XCOMET can still score
    clean_hyps = [h if h is not None else "" for h in hypotheses]

    # 4. Score
    segment_scores, system_score = score_comet(
        sources=sources,
        hypotheses=clean_hyps,
        references=references,
        batch_size=args.comet_batch,
        gpus=args.gpus,
    )

    # 5. Print summary
    print("\n" + "=" * 60)
    print(f"  Benchmark: {args.src_lang} → {args.tgt_lang}")
    print(f"  Samples  : {len(sources)}  (failed: {failed})")
    print(f"  Elapsed  : {elapsed:.1f}s  ({elapsed/len(sources)*1000:.0f} ms/sent)")
    print(f"  wmt22-comet-da system score : {system_score:.4f}")
    print(f"  wmt22-comet-da segment mean : {sum(segment_scores)/len(segment_scores):.4f}")
    print("=" * 60 + "\n")

    # 6. Save
    save_results(
        output_dir=args.output_dir,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        sources=sources,
        hypotheses=hypotheses,
        references=references,
        segment_scores=segment_scores,
        system_score=system_score,
        usages=usages,
        elapsed_s=elapsed,
        args_dict=vars(args),
    )


if __name__ == "__main__":
    # Load HF_TOKEN from .env (same pattern as inference/Hunyuan/main.py)
    _env_path = Path(__file__).parent.parent / "inference" / "Hunyuan" / ".env"
    if _env_path.exists():
        for _line in _env_path.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

    main()
