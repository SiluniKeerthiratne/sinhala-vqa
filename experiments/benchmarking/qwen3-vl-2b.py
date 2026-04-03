from huggingface_hub import login

# Login to Hugging Face
login(token="")

import nltk
try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK punkt already downloaded")
except LookupError:
    print("Downloading NLTK punkt...")
    nltk.download('punkt')

import json
import os
import zipfile
import math
import gc
import traceback
from pathlib import Path
from typing import List, Dict, Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

print("All imports successful!")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ──────────────────────────────────────────────
# Benchmark class
# ──────────────────────────────────────────────

class VLMBenchmark:
    def __init__(self, processor, model, device, checkpoint_file: str = "checkpoint.json"):
        """Initialize the benchmark with existing model and processor."""
        print("Initializing benchmark...")

        self.processor = processor
        self.model = model
        self.device = device
        self.checkpoint_file = checkpoint_file

        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=False
        )

        self.results = []
        self.errors = []
        self.processed_ids = set()

        print("✓ Benchmark initialized")

    def load_checkpoint(self):
        """Load checkpoint if it exists."""
        if os.path.exists(self.checkpoint_file):
            print(f"Loading checkpoint from {self.checkpoint_file}...")
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                self.results = checkpoint.get('results', [])
                self.errors = checkpoint.get('errors', [])
                self.processed_ids = set(checkpoint.get('processed_ids', []))
                print(f"✓ Resumed: {len(self.results)} items, {len(self.errors)} errors")
                return True
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                return False
        return False

    def save_checkpoint(self):
        """Save checkpoint after each image."""
        try:
            checkpoint = {
                'results': self.results,
                'errors': self.errors,
                'processed_ids': list(self.processed_ids)
            }
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def unzip_images(self, zip_path: str, extract_to: str):
        """Unzip the image dataset."""
        print(f"Unzipping {zip_path} to {extract_to}...")

        if os.path.exists(extract_to):
            print(f"✓ Dataset folder already exists at {extract_to}")
            return

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"✓ Successfully extracted to {extract_to}")
        except Exception as e:
            print(f"Error unzipping: {e}")
            raise

    def load_json(self, json_path: str) -> List[Dict]:
        """Load the input JSON file."""
        print(f"Loading JSON from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Loaded {len(data)} items")
        return data

    def find_image(self, image_id: int, dataset_folder: str) -> str:
        """Find image file by ID in the dataset folder."""
        extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

        for ext in extensions:
            image_path = os.path.join(dataset_folder, f"{image_id}{ext}")
            if os.path.exists(image_path):
                return image_path

            for root, dirs, files in os.walk(dataset_folder):
                image_path = os.path.join(root, f"{image_id}{ext}")
                if os.path.exists(image_path):
                    return image_path

        raise FileNotFoundError(f"Image with ID {image_id} not found in {dataset_folder}")

    def calculate_bleu(self, reference: str, hypothesis: str) -> float:
        """Calculate BLEU score."""
        try:
            reference_tokens = list(reference.strip())
            hypothesis_tokens = list(hypothesis.strip())
            smoothing = SmoothingFunction().method1
            bleu_score = sentence_bleu(
                [reference_tokens],
                hypothesis_tokens,
                smoothing_function=smoothing
            )
            return bleu_score * 100
        except Exception as e:
            print(f"Error calculating BLEU: {e}")
            return 0.0

    def calculate_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        try:
            scores = self.rouge_scorer.score(reference, hypothesis)
            return {
                "rouge1": scores['rouge1'].fmeasure * 100,
                "rouge2": scores['rouge2'].fmeasure * 100,
                "rougeL": scores['rougeL'].fmeasure * 100
            }
        except Exception as e:
            print(f"Error calculating ROUGE: {e}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    def process_single_qa(self, image_path: str, question: str, ground_truth: str,
                          qa_id: int, image_id: int) -> Dict:
        """Process a single question-answer pair."""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {
                            "type": "text",
                            "text": f"මෙම රූපය බලා පහත ප්‍රශ්නයට පිළිතුරු දෙන්න:ප්‍රශ්නය:  {question}"
                        },
                    ],
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=300)

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            generated_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            # Perplexity over generated tokens
            loss = None
            perplexity = None

            try:
                input_length = inputs.input_ids.shape[1]
                attention_mask = torch.ones_like(generated_ids)
                labels = generated_ids.clone()
                labels[:, :input_length] = -100

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=generated_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                if outputs.loss is not None:
                    loss = outputs.loss.item()
                    perplexity = math.exp(loss)
                else:
                    print(f"Warning: Loss is None for QA {qa_id}")

            except Exception as e:
                print(f"Error calculating perplexity for QA {qa_id}: {type(e).__name__}: {str(e)}")
                print(traceback.format_exc())

            bleu_score = self.calculate_bleu(ground_truth, generated_text)
            rouge_scores = self.calculate_rouge(ground_truth, generated_text)

            return {
                "answer": generated_text,
                "ground_truth": ground_truth,
                "qa_id": qa_id,
                "perplexity": perplexity,
                "loss": loss,
                "bleu_score": bleu_score,
                "rouge_scores": rouge_scores,
                "status": "success"
            }

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"Error processing QA {qa_id}: {error_msg}")
            print(f"Traceback:\n{traceback.format_exc()}")

            self.errors.append({
                "qa_id": qa_id,
                "id": image_id,
                "error": error_msg,
                "traceback": traceback.format_exc()
            })
            return {
                "answer": None,
                "ground_truth": ground_truth,
                "qa_id": qa_id,
                "perplexity": None,
                "loss": None,
                "bleu_score": None,
                "rouge_scores": None,
                "status": "error",
                "error": error_msg
            }

    def process_dataset(self, json_path: str, dataset_folder: str, save_every: int = 10):
        """Process the entire dataset."""
        data = self.load_json(json_path)

        items_to_process = [item for item in data if item["id"] not in self.processed_ids]

        if len(items_to_process) < len(data):
            print(f"Skipping {len(data) - len(items_to_process)} already processed items")

        for idx, item in enumerate(tqdm(items_to_process, desc="Processing items")):
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
                        image_id=image_id
                    )
                    result_qas.append(qa_result)

                    torch.cuda.empty_cache()
                    gc.collect()

                self.results.append({"id": image_id, "qas": result_qas})
                self.processed_ids.add(image_id)

                if (idx + 1) % save_every == 0:
                    self.save_checkpoint()

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"Error processing image {image_id}: {error_msg}")
                for qa in item["qas"]:
                    self.errors.append({
                        "qa_id": qa["qa_id"],
                        "id": image_id,
                        "error": error_msg
                    })
                self.processed_ids.add(image_id)
                self.save_checkpoint()

        self.save_checkpoint()

    def calculate_aggregate_metrics(self) -> Dict[str, float]:
        """Calculate aggregate metrics across all results."""
        all_bleu, all_rouge1, all_rouge2, all_rougeL = [], [], [], []
        all_perplexity, all_loss = [], []

        for item in self.results:
            for qa in item["qas"]:
                if qa["status"] == "success":
                    if qa["bleu_score"] is not None:
                        all_bleu.append(qa["bleu_score"])
                    if qa["rouge_scores"] is not None:
                        all_rouge1.append(qa["rouge_scores"]["rouge1"])
                        all_rouge2.append(qa["rouge_scores"]["rouge2"])
                        all_rougeL.append(qa["rouge_scores"]["rougeL"])
                    if qa["perplexity"] is not None:
                        all_perplexity.append(qa["perplexity"])
                    if qa["loss"] is not None:
                        all_loss.append(qa["loss"])

        total_qas = sum(len(item["qas"]) for item in self.results)

        return {
            "mean_bleu": sum(all_bleu) / len(all_bleu) if all_bleu else 0,
            "mean_rouge1": sum(all_rouge1) / len(all_rouge1) if all_rouge1 else 0,
            "mean_rouge2": sum(all_rouge2) / len(all_rouge2) if all_rouge2 else 0,
            "mean_rougeL": sum(all_rougeL) / len(all_rougeL) if all_rougeL else 0,
            "mean_perplexity": sum(all_perplexity) / len(all_perplexity) if all_perplexity else 0,
            "mean_loss": sum(all_loss) / len(all_loss) if all_loss else 0,
            "total_samples": len(all_bleu),
            "success_rate": len(all_bleu) / total_qas * 100 if total_qas else 0
        }

    def save_results(self, output_path: str):
        """Save results and aggregate metrics to JSON."""
        print(f"Saving results to {output_path}...")

        aggregate_metrics = self.calculate_aggregate_metrics()
        output_data = {
            "results": self.results,
            "aggregate_metrics": aggregate_metrics
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"✓ Saved {len(self.results)} results")
        print("\n=== Aggregate Metrics ===")
        print(f"Mean BLEU Score:  {aggregate_metrics['mean_bleu']:.2f}")
        print(f"Mean ROUGE-1:     {aggregate_metrics['mean_rouge1']:.2f}")
        print(f"Mean ROUGE-2:     {aggregate_metrics['mean_rouge2']:.2f}")
        print(f"Mean ROUGE-L:     {aggregate_metrics['mean_rougeL']:.2f}")
        print(f"Mean Perplexity:  {aggregate_metrics['mean_perplexity']:.4f}")
        print(f"Mean Loss:        {aggregate_metrics['mean_loss']:.4f}")
        print(f"Success Rate:     {aggregate_metrics['success_rate']:.2f}%")

    def save_errors(self, error_path: str):
        """Save errors to JSON file."""
        if self.errors:
            print(f"Saving {len(self.errors)} errors to {error_path}...")
            with open(error_path, 'w', encoding='utf-8') as f:
                json.dump(self.errors, f, ensure_ascii=False, indent=2)
        else:
            print("No errors to save!")


# ──────────────────────────────────────────────
# Model setup
# ──────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

torch.cuda.empty_cache()
gc.collect()

print("Loading Qwen3-VL model and processor...")
print("This may take a few minutes on first run...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)
print("✓ Processor loaded")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("✓ Model loaded")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✓ Using device: {device}")
print("\n=== Model Setup Complete ===")


# ──────────────────────────────────────────────
# Run benchmark
# ──────────────────────────────────────────────

INPUT_JSON      = "data/test_1000_test.json"
ZIP_FILE        = "data/filtered_images.zip"
DATASET_FOLDER  = "data/filtered_images"
OUTPUT_JSON     = "benchmarking-results-qwen3-vl-2b-instruct.json"
ERROR_JSON      = "errors.json"
CHECKPOINT_FILE = "checkpoint-qwen3-vl-2b-instruct.json"
SAVE_EVERY      = 5

benchmark = VLMBenchmark(processor, model, device, checkpoint_file=CHECKPOINT_FILE)
benchmark.load_checkpoint()

if not os.path.exists(DATASET_FOLDER):
    benchmark.unzip_images(ZIP_FILE, DATASET_FOLDER)

benchmark.process_dataset(INPUT_JSON, DATASET_FOLDER, save_every=SAVE_EVERY)
benchmark.save_results(OUTPUT_JSON)
benchmark.save_errors(ERROR_JSON)

print("\n=== Benchmark Complete ===")
print(f"Total items processed: {len(benchmark.results)}")
print(f"Total errors:          {len(benchmark.errors)}")