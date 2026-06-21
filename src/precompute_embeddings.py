"""
Run this ONCE, on your Windows machine with real internet access -- not in
my sandbox, which ran out of disk installing torch.

This is the actual "offline precompute" half of the architecture. It exists
specifically so rank.py's live step never touches sentence-transformers,
the internet, or a model load -- it just reads two .npy files. That split
is what makes the 5-minute/no-network rule survivable once you add a real
embedding model.

    pip install sentence-transformers
    python precompute_embeddings.py

Takes a few minutes on CPU for ~31K candidates -- that's fine, precompute
has no time limit. Re-run this only if features.parquet changes (i.e. you
re-ran feature_extraction.py and the candidate set or narrative_text changed).
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from jd_text import JD_TEXT


def main():
    df = pd.read_parquet("../data/precomputed/features.parquet")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")  # ~130MB, CPU-friendly

    print(f"Encoding {len(df)} candidates (CPU, no time limit here)...")
    cand_emb = model.encode(
        list(df["narrative_text"]), batch_size=64,
        show_progress_bar=True, normalize_embeddings=True,
    )
    jd_emb = model.encode([JD_TEXT], normalize_embeddings=True)

    np.save("../data/precomputed/candidate_embeddings.npy", cand_emb)
    np.save("../data/precomputed/jd_embedding.npy", jd_emb)
    df[["candidate_id"]].to_csv("../data/precomputed/embedding_ids.csv", index=False)

    print(f"Saved. Shape: {cand_emb.shape} (dim={cand_emb.shape[1]})")
    print("Don't 'git add' the .npy file directly if it crosses ~50MB -- "
          "use git-lfs, or just keep this script checked in and treat the "
          "file as regeneratable, not committed.")


if __name__ == "__main__":
    main()
