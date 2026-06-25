"""
Single-command entry point for Stage 3 reproduction.

Usage:
    python pipeline.py --candidates ./candidates.jsonl --out ./submission.csv

This runs the full ranking pipeline end-to-end:
  1. Feature extraction + hard filter  (100K -> ~31K, ~30s)
  2. BGE semantic scoring + fusion      (loads precomputed .npy, ~5s)
  3. Reasoning generator               (~5s)
  4. Reasoning patch                   (~1s)

Pre-computation note: the BGE embeddings (~48MB) are baked into this repo
(data/precomputed/*.npy). They were generated offline with
precompute_embeddings.py which takes ~5h on CPU. The ranking step above
does NOT regenerate them -- it loads them as plain numpy arrays with no
network access or model loading.

Compute constraints met:
  - CPU-only (no GPU used anywhere in this pipeline)
  - No network access (HF_HUB_OFFLINE=1 set below)
  - <5min on a 16GB CPU machine (verified: ~45s in testing)
  - <16GB RAM (peak ~2GB for 31K embeddings + feature table)
"""

import argparse
import importlib
import os
import shutil
import sys
import time

# Block any accidental HuggingFace network calls during the ranking step.
# The .npy files are already computed and baked into the repo.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


def run_step(name: str, module_name: str, t0: float):
    print(f"\n[{name}]")
    mod = importlib.import_module(module_name)
    mod.main()
    print(f"  Done — {time.time() - t0:.1f}s elapsed")


def main():
    parser = argparse.ArgumentParser(
        description="redrob-ranker: rank 100K candidates against a Senior AI Engineer JD"
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidates.jsonl (supplied by Stage 3 sandbox)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for the ranked submission CSV",
    )
    args = parser.parse_args()

    if not os.path.exists(args.candidates):
        print(f"ERROR: candidates file not found: {args.candidates}", file=sys.stderr)
        sys.exit(1)

    # Wire env vars so each module picks up the right paths
    os.environ["CANDIDATES_PATH"] = os.path.abspath(args.candidates)

    # All modules run relative to src/, so cd there
    src_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(src_dir)
    sys.path.insert(0, src_dir)

    t0 = time.time()
    print("=" * 60)
    print("redrob-ranker pipeline")
    print(f"  candidates : {args.candidates}")
    print(f"  output     : {args.out}")
    print("=" * 60)

    run_step("1/4  Feature extraction", "feature_extraction", t0)
    run_step("2/4  Ranking", "rank", t0)
    run_step("3/4  Reasoning generator", "generate_reasoning", t0)
    run_step("4/4  Reasoning patch", "patch_reasoning", t0)

    # Copy final CSV to requested output path
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    shutil.copy("../data/precomputed/submission_final.csv", args.out)

    total = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {total:.1f}s")
    print(f"Output: {args.out}")
    if total > 300:
        print(f"WARNING: exceeded 5-minute Stage 3 limit ({total:.0f}s)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
