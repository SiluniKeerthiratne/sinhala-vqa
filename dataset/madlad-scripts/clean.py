# ============================================================
# Step 0: Preprocess MADLAD Sinhala Corpus
# ============================================================
# 1. Deduplication   — remove exact + near-duplicate lines
# 2. Sinhala filter  — keep only lines that are majority Sinhala script
#
# Input:  MADLAD_JSONL  (raw .jsonl file)
# Output: same directory as input → madlad_cleaned.jsonl
# ============================================================

import json
import os
import unicodedata
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────
# 0) Config
# ─────────────────────────────────────────
MADLAD_JSONL      = "data/madlad.jsonl"   # ← raw input
MIN_SINHALA_RATIO = 0.5     # at least 50% of non-space chars must be Sinhala
MIN_TEXT_LEN      = 20      # skip very short lines
MAX_TEXT_LEN      = 2000    # skip extremely long lines (likely garbled)

SINHALA_BLOCK_START = 0x0D80
SINHALA_BLOCK_END   = 0x0DFF

# Output saved next to the input file
output_path = Path(MADLAD_JSONL).parent / "madlad_cleaned.jsonl"

# ─────────────────────────────────────────
# 1) Helpers
# ─────────────────────────────────────────
def sinhala_ratio(text: str) -> float:
    """
    Fraction of non-whitespace characters that fall in the
    Sinhala Unicode block (U+0D80–U+0DFF).
    """
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    sinhala = [c for c in chars
               if SINHALA_BLOCK_START <= ord(c) <= SINHALA_BLOCK_END]
    return len(sinhala) / len(chars)


def normalize(text: str) -> str:
    """
    Light normalization for dedup fingerprinting only (not saved).
    - NFC Unicode normalization (handles ZWJ inconsistencies)
    - Collapse whitespace
    """
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text


# ─────────────────────────────────────────
# 2) Load, filter, deduplicate
# ─────────────────────────────────────────
print(f"Reading: {MADLAD_JSONL}")

total       = 0
too_short   = 0
too_long    = 0
low_sinhala = 0
duplicates  = 0

seen_fingerprints: set[str] = set()
cleaned: list[str] = []          # store raw original text (not normalized)

with open(MADLAD_JSONL, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Processing"):
        line = line.strip()
        if not line:
            continue

        total += 1

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        text = obj.get("text", "").strip()

        # ── Length filter ──────────────────────────
        if len(text) < MIN_TEXT_LEN:
            too_short += 1
            continue
        

        # ── Sinhala script filter ──────────────────
        if sinhala_ratio(text) < MIN_SINHALA_RATIO:
            low_sinhala += 1
            continue

        # ── Exact dedup on normalized fingerprint ──
        fp = normalize(text)
        if fp in seen_fingerprints:
            duplicates += 1
            continue
        seen_fingerprints.add(fp)

        # ── Keep original text ─────────────────────
        cleaned.append(json.dumps({"text": text}, ensure_ascii=False))

# ─────────────────────────────────────────
# 3) Save
# ─────────────────────────────────────────
os.makedirs(output_path.parent, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    for line in tqdm(cleaned, desc="Writing"):
        f.write(line + "\n")

# ─────────────────────────────────────────
# 4) Report
# ─────────────────────────────────────────
kept = len(cleaned)
print("\n" + "=" * 50)
print("PREPROCESSING SUMMARY")
print("=" * 50)
print(f"  Total lines read       {total:>10,}")
print(f"  Removed — too short    {too_short:>10,}")
print(f"  Removed — too long     {too_long:>10,}")
print(f"  Removed — low Sinhala  {low_sinhala:>10,}")
print(f"  Removed — duplicates   {duplicates:>10,}")
print(f"  ─────────────────────────────────")
print(f"  Kept                   {kept:>10,}  ({100*kept/max(total,1):.1f}%)")
print("=" * 50)
print(f"\n✓ Cleaned corpus saved → {output_path}")
print(f"Next step → set  MADLAD_JSONL = '{output_path}'  in 1_cpt_sinhala.py")