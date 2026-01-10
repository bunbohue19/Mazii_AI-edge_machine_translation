import json
import os
import time
import openai
import argparse
from typing import Optional
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_SYSTEM_PROMPT = (
    "You are a Japanese interpreter and would like to translate from Japanese to other languages ​​or from other languages ​​to Japanese."
)

MAX_CONTEXT_LENGTH = 4096

def load_env_file(env_path: Path) -> None:
    """Populate os.environ entries from a simple KEY=VALUE .env file."""
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key:
            os.environ.setdefault(key.strip(), value.strip())

def _translate_text(
    client: OpenAI,
    *,
    model_name: str,
    target_lang_code: str,
    system_prompt: str,
    text: str,
    max_retries: int,
    rate_limit_backoff: float,
) -> Optional[str]:
    """Translate a single sample with retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Please translate the following segment into {target_lang_code} without any additional explanation, while fully capturing the style, nuance, and practical context of {target_lang_code}: \"{text}\"",
                    },
                ],
            )
            return completion.choices[0].message.content
        except openai.RateLimitError:
            if attempt == max_retries:
                raise
            time.sleep(rate_limit_backoff)
        except openai.BadRequestError as exc:
            print(
                "Skipping sample because the provider flagged the content as unsafe. "
                f"Details: {exc}"
            )
            return None
        except openai.OpenAIError as exc:
            print(f"OpenAI error while translating sample: {exc}")
            if attempt == max_retries:
                return None
            time.sleep(rate_limit_backoff)
    return None
        
def run_translation_job(
    *,
    text_column: str,
    dataset_path: str,
    model_name: str,
    target_lang_code: str, 
    start_id: str, 
    end_id: str,
    output_path: str
) -> None:
    """Shared runner used by every language-specific entry point."""

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    api_key = os.getenv("GEMINI_API_KEY")
    base_url = os.getenv("BASE_URL")
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    start_id = int(start_id)
    end_id = int(end_id)
    sleep_seconds = float(os.getenv("SLEEP_SECONDS", "1"))
    max_retries = int(os.getenv("MAX_RETRIES", "3"))
    rate_limit_backoff = float(os.getenv("RATE_LIMIT_BACKOFF_SECONDS", "10"))
    system_prompt = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Base")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    translated = 0

    with open(output_path, "w", encoding="utf-8") as outfile:
        outfile.write("[\n")
        for idx, sample in enumerate(tqdm(data, desc=f"Translating {dataset_path}")):
            if idx < start_id:
                continue
            if end_id is not None and idx >= end_id:
                break

            text = sample.get(text_column)

            tokens = tokenizer.tokenize(text)
            if len(tokens) >= MAX_CONTEXT_LENGTH:
                continue

            translated_text = _translate_text(
                client,
                model_name=model_name,
                target_lang_code=target_lang_code,
                system_prompt=system_prompt,
                text=text,
                max_retries=max_retries,
                rate_limit_backoff=rate_limit_backoff,
            )

            if translated_text is None:
                continue

            if translated:
                outfile.write(",\n")

            json.dump(
                {
                    "targetLanguageCode": target_lang_code,
                    "text": text,
                    "translate": translated_text,
                },
                outfile,
                ensure_ascii=False,
            )
            translated += 1
            time.sleep(sleep_seconds)

        outfile.write("\n]\n")

    print(f"Finished translating {translated} samples from {dataset_path}.")

load_env_file(Path(".env"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--model_name", type=str, default="gemini-3-flash")
    parser.add_argument("--target_lang_code", type=str, required=True)
    parser.add_argument("--start_id", type=int, required=True)
    parser.add_argument("--end_id", type=int, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
        
    run_translation_job(
        text_column=args.text_column,
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        target_lang_code=args.target_lang_code,
        start_id=args.start_id,
        end_id=args.end_id,
        output_path=args.output_path
    )
