# ============================================================
# Mixed Continual Pre-Training + VQA Fine-tuning
# Gemma 3 4B-it  |  QLoRA (4-bit nf4)  |  RTX 4090
# ============================================================
#
# Every batch = 70% MADLAD Sinhala text + 30% VQA samples
# Loss handling:
#   - text  → standard CLM loss on all tokens
#   - VQA   → loss masked to answer tokens only
#
# Directory layout expected:
#   data/
#     madlad_parquet/     ← *.parquet files (column: "text")
#     images/             ← image files named {image_id}.*
#     train-sin.json
#     test-sin.json
#
# Run:
#   python train_mixed.py
#   python train_mixed.py --resume  mixed_output/checkpoint-500
# ============================================================

import os, gc, glob, json, math, random, argparse
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)

# ──────────────────────────────────────────────────────────────
# 0) Config  ← edit these paths before running
# ──────────────────────────────────────────────────────────────
HF_TOKEN           = ""                         # or set HF_TOKEN env var
MODEL_NAME         = "google/gemma-3-4b-it"

MADLAD_JSONL       = "data/madlad_cleaned.jsonl" # JSONL file — one {"text": "..."} per line
IMAGES_DIR         = "data/images"
TRAIN_VQA_JSON     = "data/train-sin.json"
VAL_VQA_JSON       = "data/test-sin.json"

# Mixing ratio (must sum to 1.0)
TEXT_RATIO = 0.70
VQA_RATIO  = 0.30

# Training hyper-parameters
MAX_SEQ_LEN           = 512
PER_DEVICE_BATCH_SIZE = 2    # samples per forward pass
GRADIENT_ACCUMULATION = 8    # effective batch = 16
LEARNING_RATE         = 1e-5
NUM_EPOCHS            = 3
WARMUP_STEPS          = 100
VAL_STEPS             = 500  # validate every N optimizer steps
SAVE_STEPS            = 500  # checkpoint every N optimizer steps
VAL_SAMPLES           = 200  # VQA samples used for validation
SAVE_TOTAL_LIMIT      = 3    # keep this many checkpoints
SEED                  = 42

# QLoRA
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

# Prompt templates
SYSTEM_PROMPT = "ඔබ රූප දෙස බලා සිංහල භාෂාවෙන් පිළිතුරු දෙන සහායකයෙකි."
USER_PREFIX   = (
    "රූපය හොඳින් බලන්න. රූපයේ ඇති දේ පමණක් භාවිතා කර "
    "පහත ප්‍රශ්නයට සිංහලෙන් පිළිතුරු දෙන්න. "
    "එක් වචනයකින් හෝ ඉතා කෙටි පිළිතුරකින් පමණක් උත්තර දෙන්න. "
    "ප්‍රශ්නය: {question} පිළිතුර:"
)

timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"mixed_output_{timestamp}"

# ──────────────────────────────────────────────────────────────
# 1) Reproducibility + HuggingFace login
# ──────────────────────────────────────────────────────────────
random.seed(SEED)
torch.manual_seed(SEED)

token = HF_TOKEN or os.environ.get("HF_TOKEN", "")
if token:
    login(token=token)
else:
    print("⚠  No HF_TOKEN — skipping login (fine if model is cached locally)")

# ──────────────────────────────────────────────────────────────
# 2) Load 4-bit model + processor
# ──────────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

torch.cuda.empty_cache()
gc.collect()

print("Loading 4-bit Gemma 3 4B-it…")
model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

processor = AutoProcessor.from_pretrained(MODEL_NAME)
processor.tokenizer.pad_token = processor.tokenizer.eos_token

# ──────────────────────────────────────────────────────────────
# 3) Apply QLoRA
# ──────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=LORA_TARGETS,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("✓ QLoRA applied")

# ──────────────────────────────────────────────────────────────
# 4) Dataset helpers
# ──────────────────────────────────────────────────────────────

def load_madlad_texts(jsonl_path: str) -> List[str]:
    """Load raw Sinhala text strings from a JSONL file (one {"text": "..."} per line)."""
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"MADLAD JSONL not found: {jsonl_path}")
    texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading MADLAD JSONL"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", "").strip()
            if text:
                texts.append(text)
    print(f"  Loaded {len(texts):,} MADLAD documents")
    return texts


def find_image_path(image_id: int, images_dir: str) -> str:
    """Return the first matching file for a given image_id (any extension)."""
    hits = glob.glob(os.path.join(images_dir, f"{image_id}.*"))
    hits = [h for h in hits if os.path.isfile(h)]
    return hits[0] if hits else ""


def load_vqa_samples(json_path: str, images_dir: str) -> List[Dict]:
    """Load VQA pairs and resolve image paths from a JSON annotation file."""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for item in tqdm(raw, desc=f"Loading VQA: {os.path.basename(json_path)}"):
        item_id = item.get("id")
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
            a = (qa.get("answer")   or "").strip()
            if q and a:
                rows.append({"image_path": img_path, "question": q, "answer": a})
    print(f"  Loaded {len(rows):,} VQA samples from {json_path}")
    return rows


# ──────────────────────────────────────────────────────────────
# 5) MixedSinhalaDataset + MixedBatchSampler
# ──────────────────────────────────────────────────────────────

class MixedSinhalaDataset(torch.utils.data.Dataset):
    """
    Flat dataset that holds both text and VQA samples.
    Each item is tagged with 'type': 'text' | 'vqa'.
    Text items live at indices [0, n_text).
    VQA  items live at indices [n_text, n_text + n_vqa).
    """

    def __init__(self, texts: List[str], vqa_samples: List[Dict]):
        self._texts = [{"type": "text", "content": t} for t in texts]
        self._vqas  = [{"type": "vqa",  **s}           for s in vqa_samples]

    def __len__(self):
        return len(self._texts) + len(self._vqas)

    def __getitem__(self, idx: int) -> Dict:
        if idx < len(self._texts):
            return self._texts[idx]
        return self._vqas[idx - len(self._texts)]

    @property
    def n_text(self):
        return len(self._texts)

    @property
    def n_vqa(self):
        return len(self._vqas)


class MixedBatchSampler(torch.utils.data.Sampler):
    """
    Yields index lists (batches) that maintain the 70/30 text/VQA ratio.

    Each batch contains:
        n_tb = round(batch_size * text_ratio)  text indices
        n_vb = batch_size - n_tb               VQA  indices

    Text indices are in [0, n_text).
    VQA  indices are in [n_text, n_text + n_vqa).
    The number of complete batches = min(n_text // n_tb, n_vqa // n_vb).
    """

    def __init__(
        self,
        n_text: int,
        n_vqa: int,
        batch_size: int,
        text_ratio: float,
        seed: int = 42,
    ):
        self.n_text     = n_text
        self.n_vqa      = n_vqa
        self.batch_size = batch_size
        self.n_tb       = max(1, round(batch_size * text_ratio))
        self.n_vb       = batch_size - self.n_tb
        self.seed       = seed

        # How many complete batches we can form without running out of either type
        self.num_batches = min(
            n_text // self.n_tb if self.n_tb > 0 else int(1e9),
            n_vqa  // self.n_vb if self.n_vb > 0 else int(1e9),
        )

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed)

        text_idx = list(range(self.n_text))
        vqa_idx  = [self.n_text + i for i in range(self.n_vqa)]
        rng.shuffle(text_idx)
        rng.shuffle(vqa_idx)

        for b in range(self.num_batches):
            batch = (text_idx[b * self.n_tb : (b + 1) * self.n_tb] +
                     vqa_idx [b * self.n_vb : (b + 1) * self.n_vb])
            rng.shuffle(batch)   # interleave text and VQA within the batch
            yield batch


# ──────────────────────────────────────────────────────────────
# 6) Mixed Collator
# ──────────────────────────────────────────────────────────────

@dataclass
class MixedCollator:
    """
    Tokenises a mixed batch of text + VQA samples.

    Text samples  → CLM loss on ALL tokens (standard next-token prediction)
    VQA samples   → loss ONLY on answer tokens (prompt is masked with -100)

    Returns a single merged batch dict with metadata fields:
        _n_text: int  (number of text samples in this batch)
        _n_vqa:  int  (number of VQA  samples)
    These are used in the training loop to compute per-type losses.
    """

    processor: Any
    system_prompt: str
    user_prefix: str
    max_seq_len: int

    def _vqa_message_pair(self, sample: Dict) -> Tuple[List, List]:
        """Return (prompt_msgs, full_msgs) for a single VQA sample."""
        img      = Image.open(sample["image_path"]).convert("RGB")
        user_txt = self.user_prefix.format(question=sample["question"])

        sys_msg  = [{"role": "system",
                     "content": [{"type": "text", "text": self.system_prompt}]}]
        user_msg = {"role": "user",
                    "content": [{"type": "image", "image": img},
                                {"type": "text",  "text": user_txt}]}
        asst_msg = {"role": "assistant",
                    "content": [{"type": "text", "text": sample["answer"]}]}

        return sys_msg + [user_msg], sys_msg + [user_msg, asst_msg]

    def _encode_group(self, messages: List, add_generation: bool) -> Dict:
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=add_generation,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        )

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        text_samples = [s for s in batch if s["type"] == "text"]
        vqa_samples  = [s for s in batch if s["type"] == "vqa"]

        all_ids, all_masks, all_labels, all_type_ids = [], [], [], []

        # ── Text samples: loss on all tokens ─────────────────
        if text_samples:
            msgs = [[{"role": "user",
                      "content": [{"type": "text", "text": s["content"]}]}]
                    for s in text_samples]
            enc    = self._encode_group(msgs, add_generation=False)
            labels = enc["input_ids"].clone()
            pad_id = self.processor.tokenizer.pad_token_id
            if pad_id is not None:
                labels[enc["input_ids"] == pad_id] = -100
            all_ids.append(enc["input_ids"])
            all_masks.append(enc["attention_mask"])
            all_labels.append(labels)
            # text-only: no image tokens, token_type_ids are all 0
            all_type_ids.append(
                enc.get("token_type_ids", torch.zeros_like(enc["input_ids"]))
            )

        # ── VQA samples: loss on answer tokens only ───────────
        if vqa_samples:
            prompt_msgs, full_msgs = [], []
            for s in vqa_samples:
                pm, fm = self._vqa_message_pair(s)
                prompt_msgs.append(pm)
                full_msgs.append(fm)

            full   = self._encode_group(full_msgs,   add_generation=False)
            prompt = self._encode_group(prompt_msgs, add_generation=True)

            labels      = full["input_ids"].clone()
            prompt_lens = prompt["attention_mask"].sum(dim=1).tolist()
            for i, pl in enumerate(prompt_lens):
                labels[i, : int(pl)] = -100   # mask prompt + image tokens

            all_ids.append(full["input_ids"])
            all_masks.append(full["attention_mask"])
            all_labels.append(labels)
            # VQA: processor returns real token_type_ids marking image token positions
            all_type_ids.append(
                full.get("token_type_ids", torch.zeros_like(full["input_ids"]))
            )

        # ── Merge the two groups: pad to same seq length ──────
        max_len = max(t.shape[1] for t in all_ids)
        pad_id  = self.processor.tokenizer.pad_token_id or 0

        def pad_right(t: torch.Tensor, length: int, fill: int) -> torch.Tensor:
            if t.shape[1] == length:
                return t
            pad = torch.full((t.shape[0], length - t.shape[1]), fill, dtype=t.dtype)
            return torch.cat([t, pad], dim=1)

        input_ids      = torch.cat([pad_right(t, max_len, pad_id) for t in all_ids],      dim=0)
        attention_mask = torch.cat([pad_right(t, max_len, 0)      for t in all_masks],    dim=0)
        labels         = torch.cat([pad_right(t, max_len, -100)   for t in all_labels],   dim=0)
        token_type_ids = torch.cat([pad_right(t, max_len, 0)      for t in all_type_ids], dim=0)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "token_type_ids": token_type_ids,
            # Used in training loop for per-type loss computation
            "_n_text": len(text_samples),
            "_n_vqa":  len(vqa_samples),
        }


# ──────────────────────────────────────────────────────────────
# 7) Checkpoint utilities
# ──────────────────────────────────────────────────────────────

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    global_step: int,
    best_val_loss: float,
    output_dir: str,
    keep: int = 3,
):
    """Save LoRA adapter + optimizer state; prune old checkpoints."""
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)
    torch.save(
        {
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "global_step":   global_step,
            "best_val_loss": best_val_loss,
        },
        os.path.join(ckpt_dir, "trainer_state.pt"),
    )
    print(f"  ✓ Checkpoint saved → {ckpt_dir}")

    # Remove oldest checkpoints beyond the keep limit
    all_ckpts = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1]),
    )
    while len(all_ckpts) > keep:
        import shutil
        shutil.rmtree(all_ckpts.pop(0))


@torch.no_grad()
def run_validation(model, val_loader, device) -> float:
    """Compute token-weighted average loss on the VQA validation set."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in tqdm(val_loader, desc="Validation", leave=False):
        labels          = batch.pop("labels").to(device)
        input_ids       = batch.pop("input_ids").to(device)
        attention_mask  = batch.pop("attention_mask").to(device)
        token_type_ids  = batch.pop("token_type_ids").to(device)
        batch.pop("_n_text", None)
        batch.pop("_n_vqa",  None)

        out     = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels)
        n_valid = (labels != -100).sum().item()
        if n_valid > 0:
            total_loss   += out.loss.item() * n_valid
            total_tokens += n_valid

    model.train()
    return total_loss / max(total_tokens, 1)


# ──────────────────────────────────────────────────────────────
# 8) Load data
# ──────────────────────────────────────────────────────────────
print("\n── Loading datasets ──")
texts       = load_madlad_texts(MADLAD_JSONL)
train_vqa   = load_vqa_samples(TRAIN_VQA_JSON, IMAGES_DIR)
val_vqa_all = load_vqa_samples(VAL_VQA_JSON,   IMAGES_DIR)

# Fixed, shuffled validation subset
random.shuffle(val_vqa_all)
val_vqa = val_vqa_all[:VAL_SAMPLES]
print(f"  Validation subset: {len(val_vqa)} VQA samples")

# ── Training loader ───────────────────────────────────────────
train_dataset = MixedSinhalaDataset(texts, train_vqa)
train_sampler = MixedBatchSampler(
    n_text     = train_dataset.n_text,
    n_vqa      = train_dataset.n_vqa,
    batch_size = PER_DEVICE_BATCH_SIZE,
    text_ratio = TEXT_RATIO,
    seed       = SEED,
)
collator = MixedCollator(
    processor     = processor,
    system_prompt = SYSTEM_PROMPT,
    user_prefix   = USER_PREFIX,
    max_seq_len   = MAX_SEQ_LEN,
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_sampler  = train_sampler,
    collate_fn     = collator,
    num_workers    = 2,
    pin_memory     = False,    # incompatible with device_map="auto"
    prefetch_factor= 2,
)

# ── Validation loader (VQA only) ──────────────────────────────
val_dataset = MixedSinhalaDataset([], val_vqa)
val_sampler = MixedBatchSampler(
    n_text     = 0,
    n_vqa      = len(val_vqa),
    batch_size = min(2, len(val_vqa)),
    text_ratio = 0.0,
    seed       = SEED,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_sampler = val_sampler,
    collate_fn    = collator,
    num_workers   = 2,
)

batches_per_epoch = len(train_sampler)
total_opt_steps   = math.ceil(batches_per_epoch * NUM_EPOCHS / GRADIENT_ACCUMULATION)
print(f"\n  Batches / epoch      : {batches_per_epoch:,}")
print(f"  Total optimizer steps: {total_opt_steps:,}")

# ──────────────────────────────────────────────────────────────
# 9) Optimizer + LR scheduler
# ──────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps   = WARMUP_STEPS,
    num_training_steps = total_opt_steps,
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 10) Optional checkpoint resume
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default=None,
                    help="Path to a checkpoint dir to resume training from")
args = parser.parse_args()

global_step   = 0
best_val_loss = float("inf")
start_epoch   = 0

if args.resume:
    state_file = os.path.join(args.resume, "trainer_state.pt")
    if os.path.isfile(state_file):
        print(f"\nResuming from: {args.resume}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model.get_base_model(), args.resume)
        state = torch.load(state_file, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        global_step   = state["global_step"]
        best_val_loss = state["best_val_loss"]
        steps_per_epoch = math.ceil(batches_per_epoch / GRADIENT_ACCUMULATION)
        start_epoch     = global_step // steps_per_epoch
        print(f"  Resumed at optimizer step {global_step}  "
              f"(~epoch {start_epoch})  best_val_loss={best_val_loss:.4f}")
    else:
        print(f"⚠  trainer_state.pt not found in {args.resume} — starting fresh")

# ──────────────────────────────────────────────────────────────
# 11) Training loop
# ──────────────────────────────────────────────────────────────
# Determine the device that the trainable parameters live on
device = next(p for p in model.parameters() if p.requires_grad).device

log_path = os.path.join(OUTPUT_DIR, "training_log.jsonl")
log_file = open(log_path, "a", encoding="utf-8")

model.train()
optimizer.zero_grad()

print(f"\n── Starting mixed training ──────────────────────────────")
print(f"   Text ratio: {TEXT_RATIO:.0%}   VQA ratio: {VQA_RATIO:.0%}")
print(f"   Batch size (per device): {PER_DEVICE_BATCH_SIZE}")
print(f"   Gradient accumulation  : {GRADIENT_ACCUMULATION}")
print(f"   Effective batch size   : {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION}")
print(f"   Epochs: {NUM_EPOCHS}   LR: {LEARNING_RATE}")
print(f"   Output dir: {OUTPUT_DIR}\n")

for epoch in range(start_epoch, NUM_EPOCHS):
    # Different shuffle each epoch
    train_sampler.seed = SEED + epoch

    epoch_total_loss = 0.0
    # Accumulators for split logging (reset each optimizer step)
    acc_text_loss = 0.0
    acc_vqa_loss  = 0.0
    acc_micro     = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

    for micro_step, batch in enumerate(pbar):
        n_text = batch.pop("_n_text")
        n_vqa  = batch.pop("_n_vqa")

        input_ids       = batch["input_ids"].to(device)
        attention_mask  = batch["attention_mask"].to(device)
        labels          = batch["labels"].to(device)
        token_type_ids  = batch["token_type_ids"].to(device)

        # ── Forward pass ──────────────────────────────────────
        out  = model(input_ids=input_ids,
                     attention_mask=attention_mask,
                     token_type_ids=token_type_ids,
                     labels=labels)
        loss = out.loss / GRADIENT_ACCUMULATION
        loss.backward()

        epoch_total_loss += out.loss.item()

        # ── Per-type loss (approximate split, for logging only) ──
        # Text rows were stacked before VQA rows by the collator.
        with torch.no_grad():
            if n_text > 0 and n_vqa > 0:
                logits = out.logits           # (B, T, V)
                t_log  = logits[:n_text, :-1].contiguous()
                v_log  = logits[n_text:, :-1].contiguous()
                t_tgt  = labels[:n_text, 1:].contiguous()
                v_tgt  = labels[n_text:, 1:].contiguous()
                t_loss = F.cross_entropy(
                    t_log.view(-1, t_log.size(-1)), t_tgt.view(-1), ignore_index=-100)
                v_loss = F.cross_entropy(
                    v_log.view(-1, v_log.size(-1)), v_tgt.view(-1), ignore_index=-100)
                acc_text_loss += t_loss.item()
                acc_vqa_loss  += v_loss.item()
            elif n_text > 0:
                acc_text_loss += out.loss.item()
            else:
                acc_vqa_loss  += out.loss.item()
        acc_micro += 1

        # ── Optimizer step every GRADIENT_ACCUMULATION microsteps ──
        if (micro_step + 1) % GRADIENT_ACCUMULATION == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            avg_total = epoch_total_loss / (micro_step + 1)
            avg_text  = acc_text_loss / max(acc_micro, 1)
            avg_vqa   = acc_vqa_loss  / max(acc_micro, 1)
            current_lr = scheduler.get_last_lr()[0]

            pbar.set_postfix({
                "loss": f"{avg_total:.4f}",
                "txt":  f"{avg_text:.4f}",
                "vqa":  f"{avg_vqa:.4f}",
                "lr":   f"{current_lr:.2e}",
                "step": global_step,
            })

            # Write log entry
            log_entry = {
                "step":      global_step,
                "epoch":     epoch + 1,
                "loss":      round(avg_total, 5),
                "text_loss": round(avg_text,  5),
                "vqa_loss":  round(avg_vqa,   5),
                "lr":        current_lr,
            }
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

            # Reset per-step accumulators
            acc_text_loss = 0.0
            acc_vqa_loss  = 0.0
            acc_micro     = 0

            # ── Validation ────────────────────────────────────
            if global_step % VAL_STEPS == 0:
                val_loss = run_validation(model, val_loader, device)
                print(f"\n  [Step {global_step}] val_loss={val_loss:.4f}  "
                      f"(best={best_val_loss:.4f})")
                log_file.write(json.dumps({
                    "step": global_step, "val_loss": round(val_loss, 5)}) + "\n")
                log_file.flush()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Save best checkpoint
                    save_checkpoint(model, optimizer, scheduler,
                                    global_step, best_val_loss,
                                    OUTPUT_DIR, keep=SAVE_TOTAL_LIMIT)
                    with open(os.path.join(OUTPUT_DIR, "best_checkpoint.txt"), "w") as bf:
                        bf.write(f"checkpoint-{global_step}")
                    print(f"  ★  New best! Saved checkpoint-{global_step}")

            # ── Periodic save (non-best) ──────────────────────
            elif global_step % SAVE_STEPS == 0:
                save_checkpoint(model, optimizer, scheduler,
                                global_step, best_val_loss,
                                OUTPUT_DIR, keep=SAVE_TOTAL_LIMIT)

    avg_epoch = epoch_total_loss / max(len(train_loader), 1)
    print(f"\n  Epoch {epoch + 1} done — avg train loss: {avg_epoch:.4f}")

# ──────────────────────────────────────────────────────────────
# 12) Final save
# ──────────────────────────────────────────────────────────────
final_dir = os.path.join(OUTPUT_DIR, "final_adapter")
os.makedirs(final_dir, exist_ok=True)
model.save_pretrained(final_dir)
processor.save_pretrained(final_dir)
log_file.close()

print(f"\n✓  Training complete.")
print(f"   Final adapter  → {final_dir}")
print(f"   Best val loss  → {best_val_loss:.4f}")
print(f"   Training log   → {log_path}")
print(f"\nTo run inference, load base model + adapter from:\n  {final_dir}")


