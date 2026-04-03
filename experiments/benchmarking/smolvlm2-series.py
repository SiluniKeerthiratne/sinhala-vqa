"""
SmolVLM Family Benchmark
Evaluates SmolVLM2-256M, 500M, and 2.2B on the Sinhala VQA test set.

Output files per model:
  smolvlm-256m.json            smolvlm-256m-errors.json      smolvlm-256m-checkpoint.json
  smolvlm-500m.json            smolvlm-500m-errors.json      smolvlm-500m-checkpoint.json
  smolvlm-2.2b.json            smolvlm-2.2b-errors.json      smolvlm-2.2b-checkpoint.json
"""

import gc
import json
import math
import os
import traceback
import zipfile
from typing import Any, Dict, List

import torch
from bert_score import score as bertscore_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from PIL import Image
from sacrebleu.metrics import CHRF
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

# ══════════════════════════════════════════════════════════════════════
# Global config
# ══════════════════════════════════════════════════════════════════════
INPUT_JSON     = "data/test_1000_test.json"
ZIP_FILE       = "data/filtered_images.zip"
DATASET_FOLDER = "data/filtered_images"
SAVE_EVERY     = 5

MODELS = [
    ("HuggingFaceTB/SmolVLM2-256M-Video-Instruct", "smolvlm-256m"),
    ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "smolvlm-500m"),
    ("HuggingFaceTB/SmolVLM2-2.2B-Instruct",       "smolvlm-2.2b"),
]

SYSTEM_PROMPT   = "මෙම රූපය බලා පහත ප්‍රශ්නයට පිළිතුරු දෙන්න"
BERTSCORE_MODEL = "bert-base-multilingual-cased"
BERTSCORE_LANG  = "si"
BERTSCORE_BATCH = 64

_chrf = CHRF()


# ══════════════════════════════════════════════════════════════════════
# Benchmark class
# ══════════════════════════════════════════════════════════════════════
class SmolVLMBenchmark:
    def __init__(self, processor, model, device, slug: str):
        self.processor       = processor
        self.model           = model
        self.device          = device
        self.slug            = slug
        self.checkpoint_file = f"{slug}-checkpoint.json"

        self.results       : List[Dict] = []
        self.errors        : List[Dict] = []
        self.processed_ids : set        = set()

        print(f"✓ Benchmark initialised for [{slug}]")

    # ──────────────────────────────────────────────────────────────────
    # Checkpointing
    # ──────────────────────────────────────────────────────────────────
    def load_checkpoint(self) -> bool:
        if os.path.exists(self.checkpoint_file):
            print(f"Loading checkpoint from {self.checkpoint_file}...")
            try:
                with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                    ck = json.load(f)
                self.results       = ck.get("results", [])
                self.errors        = ck.get("errors", [])
                self.processed_ids = set(ck.get("processed_ids", []))
                print(f"✓ Resumed: {len(self.results)} items, {len(self.errors)} errors")
                return True
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
        return False

    def save_checkpoint(self):
        try:
            with open(self.checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "results":       self.results,
                        "errors":        self.errors,
                        "processed_ids": list(self.processed_ids),
                    },
                    f, ensure_ascii=False, indent=2,
                )
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    # ──────────────────────────────────────────────────────────────────
    # IO helpers
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def unzip_images(zip_path: str, extract_to: str):
        if os.path.exists(extract_to):
            print(f"✓ Dataset folder already exists at {extract_to}")
            return
        print(f"Unzipping {zip_path} → {extract_to}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
        print(f"✓ Extracted to {extract_to}")

    @staticmethod
    def load_json(json_path: str) -> List[Dict]:
        print(f"Loading JSON from {json_path}...")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✓ Loaded {len(data)} items")
        return data

    @staticmethod
    def find_image(image_id: int, dataset_folder: str) -> str:
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            p = os.path.join(dataset_folder, f"{image_id}{ext}")
            if os.path.exists(p):
                return p
            for root, _, _ in os.walk(dataset_folder):
                p = os.path.join(root, f"{image_id}{ext}")
                if os.path.exists(p):
                    return p
        raise FileNotFoundError(f"Image {image_id} not found in {dataset_folder}")

    # ──────────────────────────────────────────────────────────────────
    # Metrics
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def calculate_bleu(reference: str, hypothesis: str) -> float:
        try:
            return sentence_bleu(
                [list(reference.strip())],
                list(hypothesis.strip()),
                smoothing_function=SmoothingFunction().method1,
            ) * 100.0
        except Exception:
            return 0.0

    @staticmethod
    def calculate_chrf(reference: str, hypothesis: str) -> float:
        try:
            return _chrf.sentence_score(hypothesis, [reference]).score
        except Exception:
            return 0.0

    # ──────────────────────────────────────────────────────────────────
    # Core QA processing
    # ──────────────────────────────────────────────────────────────────
    def process_single_qa(
        self,
        image_path: str,
        question: str,
        ground_truth: str,
        qa_id: int,
        image_id: int,
    ) -> Dict[str, Any]:
        try:
            image = Image.open(image_path).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"{SYSTEM_PROMPT}:\nප්‍රශ්නය: {question}"},
                    ],
                }
            ]

            # Step 1: formatted prompt string (no tokenisation)
            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )

            # Step 2: tokenise text + process image
            inputs = self.processor(
                text=prompt,
                images=[image],
                return_tensors="pt",
            ).to(self.model.device)

            # Cast floats to bfloat16; leave int tensors untouched
            inputs = {
                k: v.to(dtype=torch.bfloat16) if v.dtype.is_floating_point else v
                for k, v in inputs.items()
            }

            input_len = inputs["input_ids"].shape[-1]

            # ── Generation ────────────────────────────────────────────
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=False,
                )

            generated_ids_trimmed = generated_ids[0][input_len:]
            generated_text = self.processor.decode(
                generated_ids_trimmed, skip_special_tokens=True
            ).strip()

            # ── Perplexity / loss (same approach as all Gemma exps) ───
            loss       = None
            perplexity = None
            try:
                attention_mask = torch.ones_like(generated_ids)
                labels         = generated_ids.clone()
                labels[:, :input_len] = -100

                with torch.inference_mode():
                    outputs = self.model(
                        input_ids=generated_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                if outputs.loss is not None:
                    loss       = outputs.loss.item()
                    perplexity = math.exp(loss)
                else:
                    print(f"Warning: Loss is None for QA {qa_id}")

            except Exception as e:
                print(f"Error calculating perplexity for QA {qa_id}: {type(e).__name__}: {e}")
                print(traceback.format_exc())

            bleu = self.calculate_bleu(ground_truth, generated_text)
            chrf = self.calculate_chrf(ground_truth, generated_text)

            return {
                "answer":       generated_text,
                "ground_truth": ground_truth,
                "qa_id":        qa_id,
                "bleu_score":   bleu,
                "chrF":         chrf,
                "bertscore":    None,   # filled at end of run
                "loss":         loss,
                "perplexity":   perplexity,
                "status":       "success",
            }

        except Exception as e:
            err = f"{type(e).__name__}: {str(e)}"
            print(f"Error processing QA {qa_id}: {err}")
            print(traceback.format_exc())
            self.errors.append({
                "qa_id":     qa_id,
                "id":        image_id,
                "error":     err,
                "traceback": traceback.format_exc(),
            })
            return {
                "answer":       None,
                "ground_truth": ground_truth,
                "qa_id":        qa_id,
                "bleu_score":   None,
                "chrF":         None,
                "bertscore":    None,
                "loss":         None,
                "perplexity":   None,
                "status":       "error",
                "error":        err,
            }

    def process_dataset(self, json_path: str, dataset_folder: str, save_every: int = 5):
        data = self.load_json(json_path)
        todo = [item for item in data if item["id"] not in self.processed_ids]
        if len(todo) < len(data):
            print(f"Skipping {len(data) - len(todo)} already processed items")

        for idx, item in enumerate(tqdm(todo, desc=f"[{self.slug}] items")):
            image_id = item["id"]
            try:
                image_path = self.find_image(image_id, dataset_folder)
                result_qas = []

                for qa in tqdm(item["qas"], desc=f"Image {image_id}", leave=False):
                    qa_result = self.process_single_qa(
                        image_path=image_path,
                        question=qa["question"],
                        ground_truth=qa["answer"],
                        qa_id=qa["qa_id"],
                        image_id=image_id,
                    )
                    result_qas.append(qa_result)
                    torch.cuda.empty_cache()
                    gc.collect()

                self.results.append({"id": image_id, "qas": result_qas})
                self.processed_ids.add(image_id)

                if (idx + 1) % save_every == 0:
                    self.save_checkpoint()

            except Exception as e:
                err = f"{type(e).__name__}: {str(e)}"
                print(f"Error on image {image_id}: {err}")
                for qa in item["qas"]:
                    self.errors.append({"qa_id": qa["qa_id"], "id": image_id, "error": err})
                self.processed_ids.add(image_id)
                self.save_checkpoint()

        self.save_checkpoint()

    # ──────────────────────────────────────────────────────────────────
    # BERTScore (end of run)
    # ──────────────────────────────────────────────────────────────────
    def run_bertscore(self):
        print(f"\n[{self.slug}] Computing BERTScore for all successful QAs...")
        preds, refs, ptrs = [], [], []
        for item in self.results:
            for qa in item["qas"]:
                if qa.get("status") == "success" and qa.get("answer"):
                    preds.append(qa["answer"])
                    refs.append(qa["ground_truth"])
                    ptrs.append(qa)

        if not preds:
            print("No successful samples for BERTScore.")
            return

        try:
            P, R, F1 = bertscore_score(
                cands=preds,
                refs=refs,
                lang=BERTSCORE_LANG,
                model_type=BERTSCORE_MODEL,
                verbose=True,
                batch_size=BERTSCORE_BATCH,
            )
            for qa, p, r, f in zip(ptrs, P, R, F1):
                qa["bertscore"] = {
                    "precision": float(p.item() * 100.0),
                    "recall":    float(r.item() * 100.0),
                    "f1":        float(f.item() * 100.0),
                }
            print(f"✓ BERTScore computed for {len(ptrs)} samples")
        except Exception as e:
            print(f"BERTScore error: {type(e).__name__}: {e}")
            print(traceback.format_exc())

    # ──────────────────────────────────────────────────────────────────
    # Aggregates & saving
    # ──────────────────────────────────────────────────────────────────
    def calculate_aggregate_metrics(self) -> Dict[str, float]:
        bleu_all, chrf_all                = [], []
        bs_p, bs_r, bs_f1                 = [], [], []
        perp_all, loss_all                = [], []
        total = success = 0

        for item in self.results:
            for qa in item["qas"]:
                total += 1
                if qa.get("status") != "success":
                    continue
                success += 1
                if qa.get("bleu_score") is not None:
                    bleu_all.append(qa["bleu_score"])
                if qa.get("chrF") is not None:
                    chrf_all.append(qa["chrF"])
                if qa.get("perplexity") is not None:
                    perp_all.append(qa["perplexity"])
                if qa.get("loss") is not None:
                    loss_all.append(qa["loss"])
                bs = qa.get("bertscore")
                if bs:
                    bs_p.append(bs["precision"])
                    bs_r.append(bs["recall"])
                    bs_f1.append(bs["f1"])

        def mean(xs):
            return (sum(xs) / len(xs)) if xs else 0.0

        return {
            "mean_bleu":                mean(bleu_all),
            "mean_chrF":                mean(chrf_all),
            "mean_bertscore_precision": mean(bs_p),
            "mean_bertscore_recall":    mean(bs_r),
            "mean_bertscore_f1":        mean(bs_f1),
            "mean_perplexity":          mean(perp_all),
            "mean_loss":                mean(loss_all),
            "total_samples":            success,
            "success_rate":             (success / total * 100.0) if total else 0.0,
        }

    def save_results(self, output_path: str):
        print(f"Saving results to {output_path}...")
        agg = self.calculate_aggregate_metrics()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"results": self.results, "aggregate_metrics": agg},
                      f, ensure_ascii=False, indent=2)
        print(f"✓ Saved {len(self.results)} results")
        print("\n=== Aggregate Metrics ===")
        for k, v in agg.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    def save_errors(self, error_path: str):
        if self.errors:
            print(f"Saving {len(self.errors)} errors to {error_path}...")
            with open(error_path, "w", encoding="utf-8") as f:
                json.dump(self.errors, f, ensure_ascii=False, indent=2)
        else:
            print("No errors.")


# ══════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════
def run_model(hub_id: str, slug: str):
    print(f"\n{'='*60}")
    print(f"  Model : {hub_id}")
    print(f"  Slug  : {slug}")
    print(f"{'='*60}\n")

    torch.cuda.empty_cache()
    gc.collect()

    processor = AutoProcessor.from_pretrained(hub_id)
    model = AutoModelForImageTextToText.from_pretrained(
        hub_id,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    device = next(model.parameters()).device

    benchmark = SmolVLMBenchmark(processor, model, device, slug)
    benchmark.load_checkpoint()

    if not os.path.exists(DATASET_FOLDER):
        SmolVLMBenchmark.unzip_images(ZIP_FILE, DATASET_FOLDER)

    # 1) Generation + BLEU + chrF + perplexity
    benchmark.process_dataset(INPUT_JSON, DATASET_FOLDER, save_every=SAVE_EVERY)

    # 2) BERTScore
    benchmark.run_bertscore()

    # 3) Save
    benchmark.save_results(f"{slug}.json")
    benchmark.save_errors(f"{slug}-errors.json")

    print(f"\n✓ Done [{slug}]  —  {len(benchmark.results)} items, {len(benchmark.errors)} errors")

    del model, processor, benchmark
    torch.cuda.empty_cache()
    gc.collect()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if not os.path.exists(DATASET_FOLDER):
        SmolVLMBenchmark.unzip_images(ZIP_FILE, DATASET_FOLDER)

    for hub_id, slug in MODELS:
        run_model(hub_id, slug)

    print("\n" + "=" * 60)
    print("  All SmolVLM benchmarks complete.")
    print("=" * 60)