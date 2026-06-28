# redrob-ranker

Candidate ranking system built for the India Runs Data & AI Challenge. Ranks 100K candidates against a Senior AI Engineer JD and produces a shortlist of 100, ordered by fit.

## Reproduce the submission

```bash
python src/pipeline.py --candidates ./candidates.jsonl --out ./submission.csv
```

Runtime: ~45s on CPU. Peak memory: ~2GB. No GPU, no network required.

## How it works

Five stages, each building on the last:

**1. Title filter** — An exact 47-title taxonomy classifies every candidate into Tier A (genuine ML/AI titles), Tier B (adjacent SWE/Data/DevOps), or decoy. Decoys are dropped immediately, cutting 100K to ~31K candidates.

**2. Feature extraction** — JD disqualifier rules flag consulting-only careers, title-chaser patterns, and skill self-assessment gaps. A honeypot check catches impossible YOE vs career history mismatches (>7yr gap with >2x ratio). Behavioral signals from `redrob_signals` are extracted: last-active recency, recruiter response rate, interview completion rate.

**3. Semantic scoring** — BGE (`bge-small-en-v1.5`) embeddings, precomputed offline, are loaded as numpy arrays. Cosine similarity against the JD embedding gives a semantic fit score. No model inference at ranking time.

**4. Composite fusion** — Weighted score combining:
- Title tier: 0.25
- BGE semantic fit: 0.35
- JD narrative signal count (career descriptions only): 0.10
- Skills signal coverage: 0.05
- Location fit: 0.10
- Notice period: 0.05
- Experience fit: 0.05

The composite score is multiplied by a dampened behavioral modifier (floor ~0.70x) to down-weight candidates who are strong on paper but disengaged.

**5. Reasoning generator** — Each top-100 candidate gets a 1-2 sentence explanation sourced only from their career descriptions. Company names, scale numbers, and signal hits are extracted directly from the profile — no free-form LLM generation.

## Precomputed embeddings

BGE embeddings are committed to the repo (`data/precomputed/*.npy`, ~46MB). Regenerating them takes ~5 hours on CPU:

```bash
pip install sentence-transformers
python src/precompute_embeddings.py
```

The ranking step loads the `.npy` files directly — no model, no network.

## Docker

```bash
docker build -t redrob-ranker .
```

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

[https://redrob-ranker-007.streamlit.app/](https://redrob-ranker-007.streamlit.app/)

Upload a candidate sample (≤100 candidates JSON), view ranked results with per-component score breakdown, download as CSV.

## Repository structure

```
redrob-ranker/
├── src/
│   ├── pipeline.py              # Single-command entry point
│   ├── feature_extraction.py    # Title filter + feature table
│   ├── rules.py                 # JD disqualifiers + behavioral modifier
│   ├── rank.py                  # Composite scoring + fusion
│   ├── generate_reasoning.py    # Per-candidate reasoning generator
│   ├── patch_reasoning.py       # Reasoning consistency fixes
│   ├── jd_text.py               # JD query text (single source of truth)
│   ├── precompute_embeddings.py # One-time BGE precompute (offline only)
│   └── validate_top20.py        # Hand-validation helper
├── data/
│   ├── raw/                     # candidates.jsonl (not committed, 487MB)
│   └── precomputed/
│       ├── candidate_embeddings.npy  # ~46MB, committed
│       ├── jd_embedding.npy
│       └── embedding_ids.csv
├── app.py                       # Streamlit sandbox
├── Dockerfile
├── requirements.txt
├── requirements-docker.txt
└── submission_metadata.yaml
```

## Compute constraints

| Constraint | Limit | Actual |
|---|---|---|
| Runtime | ≤5 min | ~45s |
| RAM | ≤16GB | ~2GB peak |
| GPU | None | None |
| Network at ranking time | None | None (HF_HUB_OFFLINE=1) |