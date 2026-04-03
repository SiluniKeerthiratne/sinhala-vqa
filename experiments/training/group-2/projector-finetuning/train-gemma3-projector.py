import os
import glob
import json
import gc
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import torch
from PIL import Image
from tqdm import tqdm
from datasets import Dataset

from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
    TrainingArguments,
    Trainer,
)

# -----------------------------
# 0) Login + experiment naming
# -----------------------------
login(token="")  # TODO: put token or use env var

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment = f"38k_projector_{timestamp}"

# -----------------------------
# 1) Config
# -----------------------------
MODEL_NAME = "google/gemma-3-4b-it"

IMAGES_DIR = "data/images"
TRAIN_JSON = "data/train-sin.json"
VAL_JSON   = "data/test-sin.json"

OUTPUT_DIR = "gemma3_qlora_sinhala_vqa"
ADAPTER_DIR = os.path.join(OUTPUT_DIR, f"lora_adapter_{experiment}")

SYSTEM_SI = "මෙම රූපය බලා පහත ප්‍රශ්නයට පිළිතුරු දෙන්න"

# -----------------------------
# 2) Memory cleanup
# -----------------------------
torch.cuda.empty_cache()
gc.collect()

# -----------------------------
# 3) Load 4-bit base model + processor
# -----------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("Loading 4-bit model...")
model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # Language model layers — unchanged from base experiment
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",   # LM attention
        "gate_proj", "up_proj", "down_proj",        # LM MLP
    ],
    # Projector — full precision, trained jointly with LoRA
    modules_to_save=["multi_modal_projector"],
)

model = get_peft_model(model, lora_config)
model.config.use_cache = False  # required for gradient checkpointing

# -----------------------------
# 5) Verify projector is trainable + print breakdown
# -----------------------------
print("\n" + "=" * 65)
print("  PROJECTOR EXPERIMENT — TRAINABLE PARAMETER BREAKDOWN")
print("=" * 65)

lm_lora_params        = 0
projector_params      = 0
projector_param_names = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if "multi_modal_projector" in name:
        projector_params += param.numel()
        projector_param_names.append(f"    {name}  {list(param.shape)}")
    else:
        lm_lora_params += param.numel()

total_trainable = lm_lora_params + projector_params

print(f"\n  LoRA config:")
print(f"    rank (r)       : {lora_config.r}")
print(f"    alpha          : {lora_config.lora_alpha}")
print(f"    target_modules : {lora_config.target_modules}")
print(f"    modules_to_save: {lora_config.modules_to_save}")

print(f"\n  Trainable parameter breakdown:")
print(f"    LM LoRA adapters    : {lm_lora_params:>12,}  params")
print(f"    Projector (full)    : {projector_params:>12,}  params")
print(f"    ─────────────────────────────────────────")
print(f"    Total trainable     : {total_trainable:>12,}  params")

print(f"\n  Projector parameters being trained:")
for n in projector_param_names:
    print(n)

# Sanity check — if projector_params == 0, something went wrong
if projector_params == 0:
    raise RuntimeError(
        "Projector parameters not found in trainable params. "
        "Check that 'multi_modal_projector' is the correct "
        "attribute name on this version of the model. "
        "Run: [n for n, _ in model.named_modules() if 'projector' in n.lower()]"
    )

print(f"\n  ✓ Projector is trainable and will be saved with the adapter.")
model.print_trainable_parameters()
print("=" * 65 + "\n")

# -----------------------------
# 6) Dataset helpers
# -----------------------------
def find_image_path(image_id: int, images_dir: str) -> str:
    hits = glob.glob(os.path.join(images_dir, f"{image_id}.*"))
    hits = [h for h in hits if os.path.isfile(h)]
    return hits[0] if hits else ""


def build_dataset(json_path: str, images_dir: str) -> Dataset:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows: List[Dict[str, str]] = []

    for item in tqdm(raw, desc=f"Building dataset: {json_path}"):
        item_id = item.get("id", None)

        for qa in item.get("qas", []):
            image_id = qa.get("image_id", item_id)
            if image_id is None:
                continue
            try:
                image_id = int(image_id)
            except Exception:
                continue

            img_path = find_image_path(image_id, images_dir)
            if not img_path:
                continue

            q = (qa.get("question") or "").strip()
            a = (qa.get("answer") or "").strip()
            if not q or not a:
                continue

            rows.append({"image_path": img_path, "question": q, "answer": a})

    if not rows:
        raise ValueError(
            f"No valid rows found for {json_path}. "
            "Check IMAGES_DIR and JSON format."
        )

    return Dataset.from_list(rows)


train_ds = build_dataset(TRAIN_JSON, IMAGES_DIR)
eval_ds  = build_dataset(VAL_JSON, IMAGES_DIR)

print("Train size:", len(train_ds))
print("Val size:", len(eval_ds))
print("Sample:", train_ds[0])

# -----------------------------
# 7) Data collator (loss only on assistant answer)
# -----------------------------
@dataclass
class ChatTemplateVQACollator:
    processor: Any
    system_instruction: str

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompt_messages = []
        full_messages = []

        for ex in batch:
            img = Image.open(ex["image_path"]).convert("RGB")
            q = ex["question"]
            a = ex["answer"]

            prompt_messages.append([
                {"role": "system", "content": [{"type": "text", "text": self.system_instruction}]},
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": q},
                ]},
            ])

            full_messages.append([
                {"role": "system", "content": [{"type": "text", "text": self.system_instruction}]},
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": q},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": a}]},
            ])

        # Tokenize full conversation (includes assistant answer)
        full = self.processor.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
        )

        # Tokenize prompt-only to find where assistant answer starts
        prompt = self.processor.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
        )

        input_ids = full["input_ids"]
        labels = input_ids.clone()

        # Mask everything before assistant answer — loss only on answer tokens
        prompt_lens = prompt["attention_mask"].sum(dim=1).tolist()
        for i, pl in enumerate(prompt_lens):
            labels[i, :pl] = -100

        full["labels"] = labels
        return full


collator = ChatTemplateVQACollator(
    processor=processor,
    system_instruction=SYSTEM_SI,
)

# -----------------------------
# 8) Training args
# -----------------------------
# Note: the projector parameters are float32 inside a bf16 model.
# HuggingFace Trainer handles this correctly — it will apply
# bf16=True only to the parts of the model that support it,
# and the projector's full-precision params will update in float32.
# No special configuration is needed.
#
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-5,
    num_train_epochs=2,
    warmup_ratio=0.03,
    logging_steps=10,
    save_steps=50,       # kept in sync with eval_steps so every
    eval_steps=50,       # evaluation can potentially be a checkpoint
    eval_strategy="steps",
    save_strategy="steps",
    weight_decay=0.01,   # penalises large projector weights, limits
                         # how far the projector drifts from pretrained
                         # values; has minimal effect on small LoRA matrices
    bf16=True,
    report_to="none",
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,  # keeps only the 2 best checkpoints on disk;
                         # older checkpoints are deleted automatically
                         # so you don't fill up Vast.ai storage
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
)

# -----------------------------
# 9) Train + save adapter
# -----------------------------
# The saved adapter directory will contain:
#   adapter_config.json          — LoRA config including modules_to_save
#   adapter_model.safetensors    — LoRA weight deltas (A/B matrices)
#   multi_modal_projector/       — Full projector weights (float32)
#
# To load for inference:
#   base = Gemma3ForConditionalGeneration.from_pretrained(MODEL_NAME, ...)
#   model = PeftModel.from_pretrained(base, ADAPTER_DIR)
#
trainer.train()

os.makedirs(ADAPTER_DIR, exist_ok=True)
model.save_pretrained(ADAPTER_DIR)
processor.save_pretrained(ADAPTER_DIR)

print("Saved adapter + projector to:", ADAPTER_DIR)