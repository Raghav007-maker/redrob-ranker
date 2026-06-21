"""
Day 3: semantic scoring + fusion + first top-100 CSV.

Two interchangeable semantic backends:
  - tfidf_scores()  TESTED. Pure scikit-learn, zero extra disk/network.
                    Not real semantic similarity (no synonym/paraphrase
                    understanding) -- a stand-in that proves the fusion
                    pipeline works end to end on real data, today.
  - bge_scores()    NOT TESTED HERE. Sandbox ran out of disk installing
                    torch. This is the real version -- run it on your
                    Windows machine where you have actual internet access,
                    confirm it works, then swap it in inside main().

Everything downstream of the semantic score (fusion math, sorting, CSV
write) IS tested against the real 100K pool regardless of which backend
produced semantic_score.
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from jd_text import JD_TEXT


def tfidf_scores(narrative_texts: pd.Series, jd_text: str) -> np.ndarray:
    corpus = list(narrative_texts) + [jd_text]
    vec = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    matrix = vec.fit_transform(corpus)
    sims = cosine_similarity(matrix[:-1], matrix[-1]).flatten()
    return sims


def precomputed_bge_scores(df: pd.DataFrame):
    """Loads output from precompute_embeddings.py. Returns None (not an
    error) if that hasn't been run yet, so main() can fall back to TF-IDF
    instead of crashing. This is the function the live ranking step
    actually uses -- no model load, no network, just two .npy reads."""
    emb_path = "../data/precomputed/candidate_embeddings.npy"
    jd_path = "../data/precomputed/jd_embedding.npy"
    ids_path = "../data/precomputed/embedding_ids.csv"
    if not (os.path.exists(emb_path) and os.path.exists(jd_path) and os.path.exists(ids_path)):
        return None

    cand_emb = np.load(emb_path)
    jd_emb = np.load(jd_path)
    ids = pd.read_csv(ids_path)["candidate_id"].reset_index(drop=True)

    if not (ids == df["candidate_id"].reset_index(drop=True)).all():
        raise ValueError(
            "embedding_ids.csv order doesn't match features.parquet order. "
            "Re-run feature_extraction.py then precompute_embeddings.py, in "
            "that order, without editing rows in between."
        )
    return cosine_similarity(cand_emb, jd_emb).flatten()


def composite_score(df: pd.DataFrame) -> pd.Series:
    """Hand-tuned weights -- first cut. The organizers' own example
    methodology (in submission_metadata_template.yaml) is the same shape:
    weighted components + a multiplicative behavioral modifier. Tune these
    numbers after Day 4's hand-validation pass, don't trust them blind."""
    title_bonus = df["title_tier"].map({"A": 1.0, "B": 0.4}).fillna(0.0)

    base = (
        0.45 * title_bonus
        + 0.35 * df["semantic_score"]
        + 0.10 * df["location_fit"]
        + 0.10 * df["notice_fit"]
    )
    penalty = (
        0.30 * df["flag_consulting_only"].astype(float)
        + 0.15 * df["flag_title_chaser"].astype(float)
        + 0.10 * df["flag_skill_inflation"].astype(float)
    )
    return (base - penalty).clip(lower=0) * df["behavioral_modifier"]


def main():
    df = pd.read_parquet("../data/precomputed/features.parquet")

    precomputed = precomputed_bge_scores(df)
    if precomputed is not None:
        raw = precomputed
        print("Using precomputed BGE embeddings (real semantic score).")
    else:
        raw = tfidf_scores(df["narrative_text"], JD_TEXT)
        print("No precomputed embeddings found -- using TF-IDF stand-in. "
              "Run precompute_embeddings.py first for the real thing.")

    df["semantic_score"] = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

    df["score"] = composite_score(df)
    df = df.sort_values(["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)

    top100 = df.head(100).copy()
    top100["rank"] = range(1, len(top100) + 1)
    top100["reasoning"] = (
        top100["current_title"] + " (" + top100["title_tier"] + "-tier), "
        + top100["years_of_experience"].astype(str) + " yrs, semantic fit "
        + top100["semantic_score"].round(2).astype(str)
        + ". [PLACEHOLDER -- Day 5 builds the real grounded reasoning generator]"
    )

    out = top100[["candidate_id", "rank", "score", "reasoning"]].copy()
    out["score"] = out["score"].round(4)
    out.to_csv("../data/precomputed/submission_draft.csv", index=False)

    print(f"Wrote {len(out)} rows to submission_draft.csv\n")
    print("Title tier in top 100:", top100["title_tier"].value_counts().to_dict())
    print("Top 10 preview:")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
