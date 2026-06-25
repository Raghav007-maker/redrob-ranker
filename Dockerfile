# redrob-ranker Dockerfile
# Stage 3 single-command reproduction:
#   docker build -t redrob-ranker .
#   docker run --rm --network none \
#     -v /path/to/candidates.jsonl:/data/candidates.jsonl:ro \
#     -v /path/to/output:/output \
#     redrob-ranker \
#     --candidates /data/candidates.jsonl --out /output/submission.csv
#
# Expected runtime: ~45s on CPU (tested locally)
# Memory peak:      ~2GB
# Network:          None required (embeddings baked into image)

FROM python:3.11-slim

# Block all HuggingFace network calls -- rank.py loads plain .npy files,
# no model inference happens inside the timed ranking step.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install only the ranking pipeline dependencies (no sentence-transformers,
# no streamlit, no lightgbm -- those are dev/precompute only)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Source code
COPY src/ ./src/

# Precomputed BGE embeddings baked into the image.
# Generated offline by precompute_embeddings.py (~5h on CPU).
# rank.py loads these as plain numpy arrays -- no model, no network.
COPY data/precomputed/candidate_embeddings.npy ./data/precomputed/
COPY data/precomputed/jd_embedding.npy         ./data/precomputed/
COPY data/precomputed/embedding_ids.csv        ./data/precomputed/

# Cross-encoder model baked into image (~86MB).
# Generated offline by precompute_crossencoder.py (downloads from HuggingFace once).
# rank.py loads this at runtime -- no download, no network.
COPY data/precomputed/cross_encoder_model      ./data/precomputed/cross_encoder_model

# candidates.jsonl is NOT baked in -- it is volume-mounted at runtime.
# Output goes to /output/ which should also be volume-mounted.
RUN mkdir -p data/raw

ENTRYPOINT ["python", "src/pipeline.py"]