#!/usr/bin/env python
"""Fine-tune NuExtract-tiny-v1.5 with LoRA on the ingredient parsing dataset.

Pipeline: load base model -> apply LoRA -> train with SFTTrainer -> merge -> save -> convert to GGUF.

Usage:
    uv sync --group train
    uv run python scripts/finetune.py                        # defaults
    uv run python scripts/finetune.py --epochs 5 --lr 1e-4   # tune
    uv run python scripts/finetune.py --skip-convert          # train only, no GGUF
"""

from __future__ import annotations

import argparse
import logging
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from scripts.training_data import load_training_data, rows_to_dataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "numind/NuExtract-1.5-tiny"
DATASET_PATH = PROJECT_ROOT / "tests" / "integration" / "ingredients.csv"
TMP_DIR = PROJECT_ROOT / ".tmp"
OUTPUT_DIR = TMP_DIR / "finetune-output"
LLAMA_CPP_DIR = TMP_DIR / "llama.cpp"
MODELS_DIR = PROJECT_ROOT / "models"
GGUF_NAME = "nuextract-1.5-tiny-finetuned-q8_0.gguf"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune NuExtract-tiny-v1.5 with LoRA")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--skip-convert", action="store_true", help="Skip GGUF conversion")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--gguf-output", type=Path, default=MODELS_DIR / GGUF_NAME)
    return p.parse_args()


def load_dataset_from_csv(path: Path) -> Dataset:
    return Dataset.from_list(rows_to_dataset(load_training_data(path)))


def format_chat(example: dict, tokenizer: AutoTokenizer) -> dict:
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    return {"text": text}


def train(args: argparse.Namespace) -> Path:
    logger.info("Loading base model: %s", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Graviton3+ (aarch64) and CUDA GPUs support native BF16
    use_bf16 = torch.cuda.is_available() or platform.machine() == "aarch64"
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    logger.info("Using dtype: %s", dtype)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    logger.info("Loading dataset from %s", DATASET_PATH)
    dataset = load_dataset_from_csv(DATASET_PATH)
    dataset = dataset.map(lambda ex: format_chat(ex, tokenizer), remove_columns=dataset.column_names)
    logger.info("Dataset size: %d examples", len(dataset))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=1,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=False,
        dataloader_pin_memory=torch.cuda.is_available(),
        optim="adamw_torch",
        report_to="none",
        max_grad_norm=1.0,
        max_length=args.max_seq_length,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Merging LoRA adapters...")
    merged_model = model.merge_and_unload()

    merged_path = output_dir / "merged"
    logger.info("Saving merged model to %s", merged_path)
    merged_model.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))

    return merged_path


def convert_to_gguf(merged_path: Path, gguf_output: Path) -> Path:
    if not LLAMA_CPP_DIR.exists():
        logger.info("Cloning llama.cpp to %s", LLAMA_CPP_DIR)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", str(LLAMA_CPP_DIR)],
            check=True,
        )
    else:
        logger.info("Using existing llama.cpp at %s", LLAMA_CPP_DIR)

    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found at {convert_script}")

    gguf_output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Converting to GGUF (q8_0)...")
    subprocess.run(
        [sys.executable, str(convert_script), str(merged_path), "--outfile", str(gguf_output), "--outtype", "q8_0"],
        check=True,
    )

    logger.info("GGUF saved to %s", gguf_output)
    return gguf_output


def main() -> None:
    args = parse_args()
    merged_path = train(args)

    if args.skip_convert:
        logger.info("Skipping GGUF conversion (--skip-convert)")
        logger.info("Merged model at: %s", merged_path)
        return

    gguf_path = convert_to_gguf(merged_path, args.gguf_output)
    logger.info("Done! GGUF model: %s", gguf_path)
    logger.info("Test with: MODEL_INGREDIENT_EXTRACTOR=%s uv run pytest tests/integration/ -v", gguf_path)


if __name__ == "__main__":
    main()
