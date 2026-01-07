import os
import json
import torch
import pandas as pd
from pathlib import Path
from datasets import Dataset
from unsloth import FastLanguageModel
from datetime import datetime
from trl import SFTConfig, SFTTrainer
from huggingface_hub import login

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SYSTEM_PROMPT = "You are a Japanese interpreter and would like to translate from Japanese to other languages ​​or from other languages ​​to Japanese."
USER_PROMPT = "Please translate the following segment into {target_lang_code} without any additional explanation, while fully capturing the style, nuance, and practical context of {target_lang_code}: \"{text}\""

MAX_SEQ_LENGTH = 4096
DTYPE = torch.bfloat16
LOAD_IN_4BIT = False

MODEL_ID = os.getenv("MODEL_ID")
DATASET_NAME = os.getenv("DATASET_NAME")
DATASET_PATH = f"{PROJECT_ROOT}/data/{DATASET_NAME}.json"

TIME = datetime.now().strftime("%H-%M-%S_%d-%m-%Y")
RUN_NAME = f"Mazii-MT-{TIME}"

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

def create_conversation(sample):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},                    
            {"role": "user", "content": USER_PROMPT.format(
                target_lang_code=sample['targetLanguageCode'],
                text=sample['text']
            )},
            {"role": "assistant", "content": sample["translate"]}
        ]
    }

# Define formatting function for chat template
def formatting_func(examples):
    """Convert messages to chat template format
    This function handles both single examples and batches
    """
    # Check if we're dealing with a batch or single example
    if isinstance(examples["messages"][0], list):
        # Batch processing: examples["messages"] is a list of message lists
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        return texts
    else:
        # Single example: examples["messages"] is a single message list
        text = tokenizer.apply_chat_template(
            examples["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        return [text]

load_env_file(PROJECT_ROOT / ".env")
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise RuntimeError("HF_TOKEN is not set in the environment or .env file.")
login(token=hf_token)

if __name__ == "__main__":
    df = pd.read_json(DATASET_PATH)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.shuffle()
    dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)
    dataset = dataset.train_test_split(test_size=1e-2 * len(df) / len(df))      # 1% dataset for validation
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,  
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        device_map="auto"
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",     # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )


    # Training arguments
    training_args = SFTConfig(
        output_dir=f"{PROJECT_ROOT}/output/{RUN_NAME}",
        run_name=RUN_NAME,
        logging_dir=f"{PROJECT_ROOT}/logs",
        logging_steps=1,
        logging_strategy="steps",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=4,
        # ddp_find_unused_parameters=False,  # Distributed training settings
        # ddp_timeout=3600,                  # Distributed training settings. 1 hour timeout for distributed operations
        num_train_epochs=2,
        max_steps=-1,                      # -1 means train for full epochs
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        weight_decay=0.01,
        optim="adamw_torch_fused",         
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,                        
        fp16=False,                        
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",         # Will be handled by formatting function
        packing=False,                     # Don't pack multiple samples together for translation
        seed=3407,
        report_to="tensorboard",           
    )
    
    # SFT Trainer with chat template formatting
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        formatting_func=formatting_func,
        data_collator=None,    # Default collator will be used
    )
    
    # Training, the model will be automatically saved to the Hub and the output directory
    trainer.train()
    
    # Save the final model again to the Hugging Face Hub
    trainer.save_model()
    
    # Push to hub
    trainer.push_to_hub()
