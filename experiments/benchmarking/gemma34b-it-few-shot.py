
"""
eval_fewshot_base.py
---------------------
Few-shot inference experiment on the BASE model (no adapters).
Runs 0-shot and 3-shot configs and saves separate result JSONs per config.
Uses simple prompt: "මෙම රූපය බලා පහත ප්‍රශ්නයට පිළිතුරු දෙන්න"
"""

import os
import gc
import json
import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from huggingface_hub import login

from bert_score import score as bertscore_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------

class FewShotBaseBenchmark:
    """
    Evaluation benchmark for the base model (no adapters) with configurable
    few-shot prompting. Pass few_shot_examples=[] for 0-shot.
    """

    def __init__(
        self,
        processor: Any,
        model: torch.nn.Module,
        device: torch.device,
        few_shot_examples: List[Dict],   # [{"question", "answer", "image_path"}]
        checkpoint_file: str = "checkpoint.json",
        bertscore_model_type: str = "bert-base-multilingual-cased",
        bertscore_lang: str = "si",
    ):
        self.processor = processor
        self.model = model
        self.device = device
        self.few_shot_examples = few_shot_examples
        self.checkpoint_file = checkpoint_file
        self.bertscore_model_type = bertscore_model_type
        self.bertscore_lang = bertscore_lang

        self.results: List[Dict] = []
        self.errors: List[Dict] = []
        self.processed_ids: set = set()
        self._bertscore_done: bool = False

        print(f"✓ FewShotBaseBenchmark initialised ({len(few_shot_examples)}-shot)")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def load_checkpoint(self) -> bool:
        if not os.path.exists(self.checkpoint_file):
            return False
        print(f"Loading checkpoint from {self.checkpoint_file}...")
        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
            self.results       = ckpt.get("results", [])
            self.errors        = ckpt.get("errors", [])
            self.processed_ids = set(ckpt.get("processed_ids", []))
            self._bertscore_done = bool(ckpt.get("bertscore_done", False))
            print(f"✓ Resumed: {len(self.results)} images, {len(self.errors)} errors")
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

    def save_checkpoint(self) -> None:
        try:
            ckpt = {
                "results":        self.results,
                "errors":         self.errors,
                "processed_ids":  list(self.processed_ids),
                "bertscore_done": self._bertscore_done,
                "few_shot_n":     len(self.few_shot_examples),
            }
            with open(self.checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(ckpt, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    # ------------------------------------------------------------------
    # Data utilities
    # ------------------------------------------------------------------

    def load_json(self, json_path: str) -> List[Dict]:
        print(f"Loading JSON from {json_path}...")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✓ Loaded {len(data)} items")
        return data

    def find_image(self, image_id: int, dataset_folder: str) -> str:
        extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
        for ext in extensions:
            p = os.path.join(dataset_folder, f"{image_id}{ext}")
            if os.path.exists(p):
                return p
        for root, _, _ in os.walk(dataset_folder):
            for ext in extensions:
                p = os.path.join(root, f"{image_id}{ext}")
                if os.path.exists(p):
                    return p
        raise FileNotFoundError(f"Image {image_id} not found in {dataset_folder}")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def calculate_bleu(self, reference: str, hypothesis: str) -> float:
        try:
            ref_tokens = (reference or "").strip().split()
            hyp_tokens = (hypothesis or "").strip().split()
            smoothing  = SmoothingFunction().method1
            return float(sentence_bleu([ref_tokens], hyp_tokens,
                                       smoothing_function=smoothing) * 100.0)
        except Exception as e:
            print(f"BLEU error: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    def _build_messages(self, image: Image.Image, question: str) -> List[Dict]:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text",
                              "text": "මෙම රූපය බලා පහත ප්‍රශ්නයට පිළිතුරු දෙන්න"}],
            }
        ]

        # ── Few-shot turns (image + question → answer) ──────────────────
        for ex in self.few_shot_examples:
            ex_image = Image.open(ex["image_path"]).convert("RGB")
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": ex_image},
                    {"type": "text",  "text": ex["question"]},
                ],
            })
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": ex["answer"]}],
            })

        # ── Actual query turn ────────────────────────────────────────────
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": question},
            ],
        })

        return messages

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _generate_answer_and_nll(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 128,
    ) -> Tuple[str, Optional[float], Optional[float], int, int]:
        """
        Returns: answer_text, loss, perplexity, input_len, answer_len
        """
        messages = self._build_messages(image, question)

        # Move to device — do NOT cast whole dict to bfloat16,
        # that corrupts integer input_ids
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        # Gemma-3 requires token_type_ids
        if "token_type_ids" not in inputs:
            inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])

        input_len = int(inputs["input_ids"].shape[-1])

        gen = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        answer_ids  = gen[0][input_len:]
        answer_text = self.processor.decode(answer_ids,
                                            skip_special_tokens=True).strip()
        answer_len  = int(answer_ids.shape[0])

        if answer_len == 0:
            return answer_text, None, None, input_len, 0

        # ── NLL over generated tokens only ──────────────────────────────
        full_ids       = gen
        attention_mask = torch.ones_like(full_ids, device=full_ids.device)
        token_type_ids = torch.zeros_like(full_ids, device=full_ids.device)
        labels         = full_ids.clone()
        labels[:, :input_len] = -100          # mask prompt

        outputs = self.model(
            input_ids=full_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

        loss = outputs.loss.item() if outputs.loss is not None else None
        ppl  = float(math.exp(loss)) if loss is not None else None

        return answer_text, loss, ppl, input_len, answer_len

    # ------------------------------------------------------------------
    # Per-QA processing
    # ------------------------------------------------------------------

    def process_single_qa(
        self,
        image_path: str,
        question: str,
        ground_truth: str,
        qa_id: int,
        image_id: int,
        max_new_tokens: int = 128,
    ) -> Dict:
        try:
            img = Image.open(image_path).convert("RGB")
            answer_text, loss, ppl, input_len, ans_len = self._generate_answer_and_nll(
                image=img,
                question=question,
                max_new_tokens=max_new_tokens,
            )
            bleu = self.calculate_bleu(ground_truth, answer_text)

            return {
                "qa_id":             qa_id,
                "id":                image_id,
                "question":          question,
                "ground_truth":      ground_truth,
                "answer":            answer_text,
                "loss":              loss,
                "perplexity":        ppl,
                "bleu":              bleu,
                "bertscore":         None,
                "input_len_tokens":  input_len,
                "answer_len_tokens": ans_len,
                "few_shot_n":        len(self.few_shot_examples),
                "status":            "success",
            }

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"Error QA {qa_id} (image {image_id}): {error_msg}")
            print(traceback.format_exc())
            self.errors.append({
                "qa_id": qa_id, "id": image_id,
                "error": error_msg, "traceback": traceback.format_exc(),
            })
            return {
                "qa_id": qa_id, "id": image_id,
                "question": question, "ground_truth": ground_truth,
                "answer": None, "loss": None, "perplexity": None,
                "bleu": None, "bertscore": None,
                "few_shot_n": len(self.few_shot_examples),
                "status": "error", "error": error_msg,
            }

    def process_dataset(
        self,
        json_path: str,
        dataset_folder: str,
        save_every_images: int = 10,
        max_new_tokens: int = 128,
    ) -> None:
        data = self.load_json(json_path)
        items_to_process = [x for x in data if x["id"] not in self.processed_ids]
        skipped = len(data) - len(items_to_process)
        if skipped:
            print(f"Skipping {skipped} already-processed images")

        for idx, item in enumerate(tqdm(items_to_process, desc="Processing images")):
            image_id = item["id"]
            try:
                image_path = self.find_image(image_id, dataset_folder)
                qas_out = []
                for qa in tqdm(item["qas"], desc=f"Image {image_id}", leave=False):
                    qas_out.append(self.process_single_qa(
                        image_path=image_path,
                        question=qa["question"],
                        ground_truth=qa["answer"],
                        qa_id=qa["qa_id"],
                        image_id=image_id,
                        max_new_tokens=max_new_tokens,
                    ))
                    torch.cuda.empty_cache()
                    gc.collect()

                self.results.append({"id": image_id, "qas": qas_out})
                self.processed_ids.add(image_id)

                if (idx + 1) % save_every_images == 0:
                    self.save_checkpoint()

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"Error image {image_id}: {error_msg}")
                for qa in item.get("qas", []):
                    self.errors.append({"qa_id": qa.get("qa_id"),
                                        "id": image_id, "error": error_msg})
                self.processed_ids.add(image_id)
                self.save_checkpoint()

        self.save_checkpoint()

    # ------------------------------------------------------------------
    # Batch BERTScore
    # ------------------------------------------------------------------

    def run_bert(self, batch_size: int = 32, overwrite: bool = False) -> Dict[str, float]:
        refs, hyps, idx_map = [], [], []

        for i, item in enumerate(self.results):
            for j, qa in enumerate(item.get("qas", [])):
                if qa.get("status") != "success" or qa.get("answer") is None:
                    continue
                if (not overwrite) and qa.get("bertscore") is not None:
                    continue
                refs.append(qa.get("ground_truth", ""))
                hyps.append(qa.get("answer", ""))
                idx_map.append((i, j))

        if not hyps:
            print("BERTScore: nothing to compute (already done or empty).")
            self._bertscore_done = True
            self.save_checkpoint()
            return {"mean_bertscore_precision": 0.0,
                    "mean_bertscore_recall":    0.0,
                    "mean_bertscore_f1":        0.0}

        print(f"Running BERTScore on {len(hyps)} QAs (batch_size={batch_size})...")
        P, R, F1 = bertscore_score(
            cands=hyps, refs=refs,
            model_type=self.bertscore_model_type,
            lang=self.bertscore_lang,
            device=str(self.device),
            batch_size=batch_size,
            verbose=True,
        )

        for k, (i, j) in enumerate(idx_map):
            self.results[i]["qas"][j]["bertscore"] = {
                "bertscore_precision": float(P[k].item() * 100.0),
                "bertscore_recall":    float(R[k].item() * 100.0),
                "bertscore_f1":        float(F1[k].item() * 100.0),
            }

        self._bertscore_done = True
        self.save_checkpoint()

        agg = {
            "mean_bertscore_precision": float(P.mean().item() * 100.0),
            "mean_bertscore_recall":    float(R.mean().item() * 100.0),
            "mean_bertscore_f1":        float(F1.mean().item() * 100.0),
        }
        print("✓ BERTScore done:", agg)
        return agg

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    def calculate_aggregate_metrics(self) -> Dict[str, float]:
        bleus, losses, ppls = [], [], []
        bP, bR, bF = [], [], []
        total_qas = success_qas = 0

        for item in self.results:
            for qa in item["qas"]:
                total_qas += 1
                if qa.get("status") != "success":
                    continue
                success_qas += 1
                if qa.get("bleu")       is not None: bleus.append(qa["bleu"])
                if qa.get("loss")       is not None: losses.append(qa["loss"])
                if qa.get("perplexity") is not None: ppls.append(qa["perplexity"])
                if qa.get("bertscore")  is not None:
                    bP.append(qa["bertscore"]["bertscore_precision"])
                    bR.append(qa["bertscore"]["bertscore_recall"])
                    bF.append(qa["bertscore"]["bertscore_f1"])

        return {
            "few_shot_n":              len(self.few_shot_examples),
            "mean_bleu":               safe_mean(bleus),
            "mean_loss":               safe_mean(losses),
            "mean_perplexity":         safe_mean(ppls),
            "mean_bertscore_precision": safe_mean(bP),
            "mean_bertscore_recall":   safe_mean(bR),
            "mean_bertscore_f1":       safe_mean(bF),
            "total_qas":               float(total_qas),
            "success_qas":             float(success_qas),
            "success_rate":            float(success_qas / total_qas * 100.0) if total_qas else 0.0,
        }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_results(self, output_path: str) -> None:
        print(f"Saving results to {output_path}...")
        agg = self.calculate_aggregate_metrics()
        out = {"results": self.results, "aggregate_metrics": agg, "errors": self.errors}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("✓ Saved")
        print("\n=== Aggregate Metrics ===")
        for k, v in agg.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    def save_errors(self, error_path: str) -> None:
        if not self.errors:
            print("No errors to save.")
            return
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(self.errors, f, ensure_ascii=False, indent=2)
        print(f"✓ Errors saved to {error_path}")


# ===========================================================================
# Runner
# ===========================================================================

def main():

    # ── HuggingFace login ────────────────────────────────────────────────────
    login(token="")

    # ── Paths ────────────────────────────────────────────────────────────────
    BASE_MODEL  = "google/gemma-3-4b-it"
    TEST_JSON   = "data/test_1000_test.json"
    IMAGES_DIR  = "data/filtered_images"
    OUT_DIR     = "result_jsons"
    OTHER_DIR   = "other_jsons"
    os.makedirs(OUT_DIR,   exist_ok=True)
    os.makedirs(OTHER_DIR, exist_ok=True)

    MAX_NEW_TOKENS    = 128
    SAVE_EVERY_IMAGES = 10

    # ── Few-shot examples ────────────────────────────────────────────────────
    FEW_SHOT_EXAMPLES = [
        {
            "question":   "ජනේලය හදලා තියෙන්නේ මොනවා වලින්ද?",
            "answer":     "යකඩ ග්‍රිල් වලින්.",
            "image_path": "data/few_shot/2365841.jpg",
        },
        {
            "question":   "පසුබිමේ තියෙන්නෙ මොකක්ද?",
            "answer":     "බිත්තියක්.",
            "image_path": "data/few_shot/2332532.jpg",
        },
        {
            "question":   "ඒ මනුස්සයා බයිසිකලේ උඩ මොකක්ද කරන්නේ?",
            "answer":     "ඔළුවෙන් හිටගෙන ඉන්නවා.",
            "image_path": "data/few_shot/2393692.jpg",
        },
    ]

    CONFIGS = {
        "0shot_base": [],
        "3shot_base": FEW_SHOT_EXAMPLES,
    }

    # ── Load base model ONCE ─────────────────────────────────────────────────
    print("Loading base model...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL)

    device = next(model.parameters()).device
    print(f"✓ Model on {device}")

    # ── Run both configs ─────────────────────────────────────────────────────
    for run_label, examples in CONFIGS.items():
        run_id    = f"base_{run_label}"
        out_path  = os.path.join(OUT_DIR,   f"{run_id}.json")
        err_path  = os.path.join(OTHER_DIR, f"{run_id}_errors.json")
        ckpt_path = os.path.join(OTHER_DIR, f"{run_id}_checkpoint.json")

        print(f"\n{'='*60}")
        print(f"  Running: {run_id}  ({len(examples)} examples in context)")
        print(f"{'='*60}")

        torch.cuda.empty_cache()
        gc.collect()

        bench = FewShotBaseBenchmark(
            processor=processor,
            model=model,
            device=device,
            few_shot_examples=examples,
            checkpoint_file=ckpt_path,
            bertscore_model_type="bert-base-multilingual-cased",
            bertscore_lang="si",
        )

        bench.load_checkpoint()

        bench.process_dataset(
            json_path=TEST_JSON,
            dataset_folder=IMAGES_DIR,
            save_every_images=SAVE_EVERY_IMAGES,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        bench.run_bert(batch_size=32)
        bench.save_results(out_path)
        bench.save_errors(err_path)

        print(f"✓ {run_id} complete → {out_path}")

    print("\n✓ All configs complete.")


if __name__ == "__main__":
    main()