# ============================================================
# Step 1: Continual Pre-Training (CPT) on MADLAD-400 Sinhala
# ============================================================
# Dataset: local JSONL file — Format: {"text": "..."}
#
# Uses Gemma3ForConditionalGeneration + AutoProcessor
# (same stack as VQA scripts) so token_type_ids are handled
# automatically by apply_chat_template.
#
# Each line is a full document (~840 words average).
# Documents are chunked into MAX_SEQ_LEN token windows
# with overlap so the full corpus is used efficiently.
#
# Output:  cpt_output/cpt_adapter_TIMESTAMP/
# ============================================================
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import torch
from datasets import Dataset
from tqdm import tqdm

from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
    TrainingArguments,
    Trainer,
)

# ─────────────────────────────────────────
# 0) Config
# ─────────────────────────────────────────
HF_TOKEN     = ""                              # or set HF_TOKEN env var
MODEL_NAME   = "google/gemma-3-4b-it"

MADLAD_JSONL = "data/test.jsonl"    # ← output of script 0
MAX_SEQ_LEN  = 512                             # tokens per chunk
CHUNK_STRIDE = 448                             # stride = 512 - 64 overlap
MIN_CHUNK_TOKENS = 32                          # discard tiny tail chunks
MAX_CHUNKS   = 150_000                         # None = use all (~1.5M, 25-50hrs)
SEED         = 42

timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR  = "cpt_output"
ADAPTER_DIR = os.path.join(OUTPUT_DIR, f"cpt_adapter_{timestamp}")

# ─────────────────────────────────────────
# 1) Login
# ─────────────────────────────────────────
token = HF_TOKEN or os.environ.get("HF_TOKEN", "")
if token:
    login(token=token)
else:
    print("⚠  No HF token — skipping login")

# ─────────────────────────────────────────
# 2) Load model + processor
# ─────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

torch.cuda.empty_cache()
gc.collect()

print("Loading 4-bit model...")
model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)
model.config.use_cache = False

processor = AutoProcessor.from_pretrained(MODEL_NAME)
processor.tokenizer.pad_token = processor.tokenizer.eos_token

# ─────────────────────────────────────────
# 3) Apply QLoRA
# ─────────────────────────────────────────
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("✓ QLoRA CPT model ready")

# ─────────────────────────────────────────
# 4) Chunking helper
# ─────────────────────────────────────────
def chunk_document(text: str, max_len: int, stride: int) -> list[str]:
    """
    Tokenize a full document and split into overlapping token windows.
    Returns decoded text chunks ready for apply_chat_template.
    """
    token_ids = processor.tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for start in range(0, len(token_ids), stride):
        chunk_ids = token_ids[start : start + max_len]
        if len(chunk_ids) < MIN_CHUNK_TOKENS:
            break
        chunks.append(processor.tokenizer.decode(chunk_ids, skip_special_tokens=True))
    return chunks

# ─────────────────────────────────────────
# 5) Load JSONL and chunk all documents
# ─────────────────────────────────────────
def load_and_chunk(path: str, max_chunks: int | None) -> Dataset:
    all_chunks = []
    docs_read  = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Chunking documents"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = obj.get("text", "").strip()
            if len(text) < 20:
                continue

            all_chunks.extend(chunk_document(text, MAX_SEQ_LEN, CHUNK_STRIDE))
            docs_read += 1

            if max_chunks and len(all_chunks) >= max_chunks * 2:
                break

    print(f"Documents read: {docs_read:,}  |  Total chunks: {len(all_chunks):,}")

    random.seed(SEED)
    random.shuffle(all_chunks)

    if max_chunks and len(all_chunks) > max_chunks:
        all_chunks = all_chunks[:max_chunks]
        print(f"Capped to: {len(all_chunks):,} chunks")

    return Dataset.from_list([{"text": c} for c in all_chunks])


raw_ds   = load_and_chunk(MADLAD_JSONL, MAX_CHUNKS)
split    = raw_ds.train_test_split(test_size=0.005, seed=SEED)
train_ds = split["train"]
eval_ds  = split["test"]
print(f"Train: {len(train_ds):,}  |  Val: {len(eval_ds):,}")

# ─────────────────────────────────────────
# 6) Collator — text-only, loss on all tokens
# ─────────────────────────────────────────
@dataclass
class CPTTextCollator:
    processor: Any

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Wrap each chunk as a plain user message — no images
        # apply_chat_template handles token_type_ids automatically
        messages = [
            [{"role": "user",
              "content": [{"type": "text", "text": ex["text"]}]}]
            for ex in batch
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
        )

        # For CPT, loss on ALL tokens (no masking like VQA)
        inputs["labels"] = inputs["input_ids"].clone()

        # Mask padding tokens from loss
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            inputs["labels"][inputs["input_ids"] == pad_id] = -100

        return inputs


collator = CPTTextCollator(processor=processor)

# ─────────────────────────────────────────
# 7) Training
# ─────────────────────────────────────────
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,    # effective batch = 16
    learning_rate=5e-6,
    num_train_epochs=1,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    logging_steps=50,
    save_steps=500,
    eval_steps=500,
    eval_strategy="steps",
    save_strategy="steps",
    bf16=True,
    report_to="none",
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    dataloader_num_workers=2,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
)

print(f"Starting CPT — {len(train_ds):,} chunks, 1 epoch...")
trainer.train()

# ─────────────────────────────────────────
# 8) Save CPT adapter + processor
# ─────────────────────────────────────────
os.makedirs(ADAPTER_DIR, exist_ok=True)
model.save_pretrained(ADAPTER_DIR)
processor.save_pretrained(ADAPTER_DIR)

print(f"\n✓ CPT adapter saved → {ADAPTER_DIR}")
print(f"Next step → set  CPT_ADAPTER_DIR = '{ADAPTER_DIR}'  in 2_finetune_vqa.py")