"""
Run this ONCE on your Windows machine to download the cross-encoder model.

    python precompute_crossencoder.py

Downloads ~86MB. Saves to data/precomputed/cross_encoder_model/.
Commit that folder to git -- it gets COPY'd into Docker so ranking
works fully offline (HF_HUB_OFFLINE=1 blocks any internet calls at runtime).

Why a cross-encoder vs the BGE bi-encoder we already have:
  BGE bi-encoder: embeds JD and candidates independently, then measures
  cosine similarity. Fast (precomputed), but misses cross-attention between
  JD tokens and candidate tokens.

  Cross-encoder: takes (JD, candidate) as a PAIR, runs them jointly through
  a transformer, produces a single relevance score. Slower (can't precompute
  for unknown queries) but meaningfully more accurate for the top-K rerank.
  The standard two-stage pattern: BGE recalls 100 candidates, cross-encoder
  reranks the top 40 of those 100 with high precision.

Expected timing on CPU (40 candidate pairs):
  ~20-25 seconds. Combined with the rest of the pipeline (~45s), total
  stays under 90 seconds -- well inside Stage 3's 5-minute limit.
"""

import os
from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SAVE_PATH = "../data/precomputed/cross_encoder_model"

print(f"Downloading {MODEL_NAME} (~86MB)...")
model = CrossEncoder(MODEL_NAME)
model.save(SAVE_PATH)
print(f"Saved to {SAVE_PATH}")
print("Now add this folder to git:")
print("  git add data/precomputed/cross_encoder_model/")
print("  git commit -m 'feat: add cross-encoder model for rerank step'")
