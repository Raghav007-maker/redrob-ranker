# redrob-ranker

AI recruiting system that ranks 100K candidates against a Senior AI Engineer JD.
Built for the India Runs Data & AI Challenge hackathon.

## Quick start — single command reproduction

```bash
python src/pipeline.py --candidates ./candidates.jsonl --out ./submission.csv
```

Expected runtime: ~45s on CPU. Peak memory: ~2GB.

## Architecture

Five-stage pipeline:

1. **Hard filter** — 47-title exact taxonomy cuts 100K → 31K candidates (Tier A/B, decoys excluded)
2. **Feature extraction** — JD disqualifier rules + behavioral signals from `redrob_signals`
3. **Semantic scoring** — precomputed BGE (`bge-small-en-v1.5`) cosine similarity vs JD
4. **Composite fusion** — weighted combination: title (0.25) + semantic (0.40) + JD signals (0.15) + location (0.10) + notice (0.05) + experience (0.05), multiplied by behavioral modifier
5. **Reasoning generator** — grounded 1-2 sentence explanation per candidate, sourced from career descriptions only (no hallucination)

## Pre-computation (one-time, not in the 5-minute window)

BGE embeddings are precomputed and committed to the repo (`data/precomputed/*.npy`, ~48MB).
To regenerate them:

```bash
pip install sentence-transformers
python src/precompute_embeddings.py   # ~5h on CPU
```

The ranking step (`pipeline.py`) loads these as plain numpy arrays — no model inference,
no network access at ranking time.

## Docker

Build:
```bash
docker build -t redrob-ranker .
```

Run:
```bash
# Linux/macOS
docker run --rm --network none \
  -v $(pwd)/data/raw/candidates.jsonl:/data/candidates.jsonl:ro \
  -v $(pwd)/output:/output \
  redrob-ranker \
  --candidates /data/candidates.jsonl --out /output/submission.csv

# Windows PowerShell
docker run --rm --network none `
  -v "${PWD}\data\raw\candidates.jsonl:/data/candidates.jsonl:ro" `
  -v "${PWD}\output:/output" `
  redrob-ranker `
  --candidates /data/candidates.jsonl --out /output/submission.csv
```

## Sandbox

Live demo on Streamlit Cloud: [link TBD — deploy from this repo]

Accepts a small candidate sample (≤100 candidates), runs the full pipeline,
shows score breakdown per component, allows CSV download.

## Repository structure

```
redrob-ranker/
├── src/
│   ├── pipeline.py              # Single-command entry point
│   ├── feature_extraction.py    # Title filter + feature table
│   ├── rules.py                 # JD disqualifiers + behavioral modifier
│   ├── rank.py                  # Composite scoring + fusion
│   ├── generate_reasoning.py    # Per-candidate reasoning generator
│   ├── patch_reasoning.py       # Stage 4 consistency fixes
│   ├── jd_text.py               # JD query text
│   ├── precompute_embeddings.py # One-time BGE precompute (offline)
│   └── validate_top20.py        # Day 4 hand-validation helper
├── data/
│   ├── raw/                     # candidates.jsonl (NOT committed, 487MB)
│   └── precomputed/
│       ├── candidate_embeddings.npy  # Committed (~48MB)
│       ├── jd_embedding.npy          # Committed
│       └── embedding_ids.csv         # Committed
├── app.py                       # Streamlit sandbox
├── Dockerfile
├── requirements.txt             # Full dev dependencies
├── requirements-docker.txt      # Minimal container dependencies
└── submission_metadata.yaml
```

## Compute constraints (Stage 3)

| Constraint | Limit | Actual |
|---|---|---|
| Runtime | ≤5 min | ~45s |
| RAM | ≤16GB | ~2GB peak |
| GPU | None | None used |
| Network | None | None (HF_HUB_OFFLINE=1) |
