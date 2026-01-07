import json
import os
import time
import openai
from typing import Optional
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

DEFAULT_SYSTEM_PROMPT = (
    "あなたは日本語の通訳者です。日本語から他の言語へ、または他の言語から日本語へ翻訳してください。"
)

MAX_CONTEXT_LENGTH = 4096

def _require_env(name: str) -> str:
    """Return the env var or raise a helpful error."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} environment variable is required for the translation job.")
    return value


def _parse_optional_int(value: Optional[str], default_value: Optional[int]) -> Optional[int]:
    if value is None or value == "":
        return default_value
    return int(value)


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
                        "content": f"次のセグメントを、追加の説明なしで \"{target_lang_code}\" に翻訳してください。\"{text}\"",
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
    text_column: str = "formatted",
    dataset_path: str = "source_data/[JA-JA]_大辞泉_第二版.jsonl",
    default_model_name: str = "glm-4.5-flash",
) -> None:
    """Shared runner used by every language-specific entry point."""

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    api_key = _require_env("ZAI_API_KEY")
    base_url = os.getenv("BASE_URL")
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    target_lang_code = _require_env("TARGET_LANG_CODE")
    output_fp = _require_env("OUTPUT_FP")

    start_id = _parse_optional_int(os.getenv("START_ID"), 0) or 0
    end_id = _parse_optional_int(os.getenv("END_ID"), None)
    sleep_seconds = float(os.getenv("SLEEP_SECONDS", "1"))
    max_retries = int(os.getenv("MAX_RETRIES", "3"))
    rate_limit_backoff = float(os.getenv("RATE_LIMIT_BACKOFF_SECONDS", "10"))
    model_name = os.getenv("MODEL_NAME", default_model_name)
    system_prompt = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

    tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.5")

    output_dir = os.path.dirname(output_fp)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    translated = 0

    with open(output_fp, "w", encoding="utf-8") as outfile:
        outfile.write("[\n")
        for idx, sample in enumerate(tqdm(data, desc=f"Translating {dataset_path}")):
            if idx < start_id:
                continue
            if end_id is not None and idx >= end_id:
                break

            text = sample.get(text_column)
            tokens = tokenizer.encode(text)
            if not text or len(tokens) >= MAX_CONTEXT_LENGTH:
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

if __name__ == "__main__":
    run_translation_job(dataset_path="[JA-JA]_大辞泉_第二版.jsonl")
