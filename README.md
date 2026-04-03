# Sinhala Visual Question Answering

Code and experiments for my BSc AI & Data Science Individual Research Project at Robert Gordon University.

**Thesis title:** Adapting Vision-Language Models for Sinhala Visual Question Answering

---

## Overview

This project investigates parameter-efficient fine-tuning of vision-language models (VLMs) for Sinhala, a low-resource language. The core model is Gemma-3-4B-IT fine-tuned with QLoRA on a ~36k Sinhala VQA dataset translated from Visual Genome using Gemini Flash.

Key approaches explored:
- Zero-shot and few-shot prompting baselines (SmolVLM, Qwen2.5-VL, Gemma-3)
- Direct QLoRA fine-tuning at different dataset scales
- Sequential continual pre-training (CPT) on MADLAD-400 Sinhala → VQA fine-tuning
- Mixed CPT + VQA simultaneous training

---

## Repository Structure
sinhala-vqa/
├── dataset_creation/
│   └── translation_scripts/    # Gemini Flash translation pipeline
├── experiments/
│   ├── benchmarking/           # Zero-shot and few-shot baseline scripts
│   ├── training/               # QLoRA, CPT, and mixed training scripts
│   └── evaluation/             # LLM-as-judge evaluation pipeline
├── eval_results/               # Metric outputs and result CSVs
├── configs/                    # Training configs and hyperparameters
├── notebooks/                  # Analysis and plotting notebooks
└── requirements.txt

---

## Model & Dataset

- **Base model:** `google/gemma-3-4b-it`
- **Dataset:** ~36k Sinhala VQA pairs (translated from Visual Genome)
- **CPT corpus:** MADLAD-400 Sinhala
- **Fine-tuned models:** Available on HuggingFace at [siluni](https://huggingface.co/siluni)

---

## Evaluation

Models are evaluated using a dual LLM-as-judge pipeline (Gemini 2.0 Flash + Claude Sonnet) alongside automatic metrics: BLEU, chrF, BERTScore, perplexity, foreign character ratio, and word repetition rate.

---

## Requirements
```bash
pip install -r requirements.txt
```

---

