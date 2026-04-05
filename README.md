# Sinhala Visual Question Answering

Code and experiments for my BSc AI & Data Science Individual Research Project at Robert Gordon University.

**Thesis title:** Adapting Vision-Language Models for Sinhala Visual Question Answering

## Overview

This project investigates parameter-efficient fine-tuning of vision-language models (VLMs) for Sinhala, a low-resource language. The core model is Gemma-3-4B-IT fine-tuned with QLoRA on a ~37k Sinhala VQA dataset translated from Visual Genome using Gemini Flash.

Four approaches are systematically compared:

| Approach | Description |
|----------|-------------|
| Zero-shot / Few-shot prompting | Untuned baselines across SmolVLM2, Qwen3-VL, Gemma-3 |
| Direct QLoRA fine-tuning | LoRA on attention + MLP layers at varying dataset scales |
| Projector fine-tuning | QLoRA + full-precision vision-language projector training |
| Sequential CPT → VQA | MADLAD-400 Sinhala continual pre-training, then VQA fine-tuning |
| Mixed CPT + VQA | Simultaneous training with 70/30 text-to-VQA batch ratio |

## Repository Structure

```
sinhala-vqa/
├── requirements.txt
├── dataset/
│   ├── vqa/
│   │   └── traslation-script.ipynb      # Gemini Flash translation pipeline (EN → SI)
│   └── madlad-scripts/
│       └── clean.py                     # MADLAD-400 dedup + Sinhala script filter
└── experiments/
    ├── benchmarking/
    │   ├── gemma3-4b-it.ipynb            # Gemma-3-4B-IT zero-shot (notebook)
    │   ├── gemma34b-it-few-shot.py       # Gemma-3-4B-IT few-shot
    │   ├── gemma3n-e2b.ipynb             # Gemma-3n-E2B zero-shot (notebook)
    │   ├── qwen3-vl-2b.py               # Qwen3-VL-2B benchmark
    │   ├── qwen3-vl-4b.py               # Qwen3-VL-4B benchmark
    │   └── smolvlm2-series.py           # SmolVLM2 256M / 500M / 2.2B benchmark
    └── training/
        ├── group-2/
        │   ├── projector-finetuning/
        │   │   ├── train-gemma3-projector.py   # QLoRA + projector fine-tuning
        │   │   └── test_projector.ipynb
        │   └── scaling-study/
        │       ├── training-script.py          # QLoRA at varying data scales
        │       └── test-task-fine-tunning.ipynb
        └── group-3/
            ├── mixed/
            │   ├── train_mixed.py              # Simultaneous CPT + VQA training
            │   └── test-mix-model.ipynb
            └── sequential/
                ├── train_cpt.py                # MADLAD-400 continual pre-training
                ├── test-cpt-few-shot.py        # Few-shot eval on CPT-only model
                └── test-sequential-cpt-vqa.ipynb
```

## Dataset

- **VQA pairs:** ~37k Sinhala question-answer pairs translated from Visual Genome using Gemini Flash (gemini-3-flash-preview)
- **CPT corpus:** MADLAD-400 Sinhala subset, cleaned and deduplicated
- **Fine-tuned models:** Available on HuggingFace at [siluni](https://huggingface.co/siluni)

### Expected data layout

```
data/
├── images/                  # Visual Genome images ({image_id}.jpg)
├── train-sin.json           # Training VQA pairs
├── test-sin.json            # Validation VQA pairs
└── madlad_cleaned.jsonl     # Cleaned MADLAD-400 Sinhala corpus
```

### Dataset preprocessing

The MADLAD cleaning pipeline (`dataset/madlad-scripts/clean.py`) applies:

- Exact and normalised (NFC) deduplication
- Sinhala script filter: retains only texts where >50% of characters fall in Unicode block U+0D80–U+0DFF
- Length filter: 20–2000 characters

## Models

| Model | Size | Role |
|-------|------|------|
| google/gemma-3-4b-it | 4B | Primary fine-tuning target |
| google/gemma-3n-e2b-it | 2B | Benchmarking baseline |
| Qwen/Qwen3-VL-2B-Instruct | 2B | Benchmarking baseline |
| Qwen/Qwen3-VL-4B-Instruct | 4B | Benchmarking baseline |
| HuggingFaceTB/SmolVLM2-256M-Video-Instruct | 256M | Benchmarking baseline |
| HuggingFaceTB/SmolVLM2-500M-Video-Instruct | 500M | Benchmarking baseline |
| HuggingFaceTB/SmolVLM2-2.2B-Instruct | 2.2B | Benchmarking baseline |

## Training

### Common hyperparameters

| Parameter | Value |
|-----------|-------|
| Quantization | 4-bit NF4 + double quantization |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| LoRA targets | q/k/v/o_proj, gate/up/down_proj |
| Compute dtype | bfloat16 |
| Per-device batch size | 1 |
| Gradient accumulation | 16 (effective batch = 16) |
| Learning rate (VQA) | 1e-5 |
| Learning rate (CPT) | 5e-6 |

### Projector fine-tuning (Group 2)

In addition to standard QLoRA layers, the vision-language projector (`multi_modal_projector`) is trained in full float32 precision using `modules_to_save`. Loss is masked to answer tokens only. The system prompt is in Sinhala: "මෙම රූපය බලා පහත ප්‍රශ්නයට පිළිතුරු දෙන්න".

### Sequential CPT → VQA (Group 3a)

Run `train_cpt.py` to fine-tune on MADLAD-400 Sinhala with a 512-token sliding window (448-token stride). Loss is computed over all tokens.  
Load the resulting CPT adapter, then run VQA fine-tuning with answer-only loss masking.

### Mixed CPT + VQA (Group 3b)

`train_mixed.py` trains simultaneously on MADLAD text and VQA data, maintaining a 70/30 ratio per batch via a custom `MixedBatchSampler`. Text batches use CLM loss over all tokens; VQA batches mask loss to answer tokens only. Per-type losses are logged separately.

Supports checkpoint resuming:

```bash
python train_mixed.py --resume checkpoint-500
```

## Evaluation

Models are evaluated using:

- **Automatic metrics:** BLEU, chrF, BERTScore (multilingual bert-base-multilingual-cased), ROUGE-1/2/L, perplexity
- **LLM-as-judge:** Dual-judge pipeline using gemini-3-flash-preview + Claude Sonnet 4.6 for semantic correctness scoring
- **Training diagnostics:** Per-type loss tracking (text vs. VQA), foreign character ratio, word repetition rate

## Setup

```bash
pip install -r requirements.txt
```

GPU with ≥16 GB VRAM recommended (tested on RTX 4090). 4-bit quantization enables inference on smaller GPUs.

