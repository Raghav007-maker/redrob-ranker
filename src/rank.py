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


def experience_fit(yoe: float) -> float:
    """JD targets 3-8 years of recent production ML experience.
    Under 2 years -- too junior for a Senior role.
    3-8 years -- sweet spot.
    9-12 years -- acceptable, but the JD's language skews younger/hands-on.
    13+ years -- was 0.5x, loosened to 0.65x. CAND_0039754 (Meta + Apple,
    15 JD signal hits) showed the 0.5x penalty was burying genuinely modern
    candidates who just happen to have long careers."""
    if yoe < 2:
        return 0.4
    elif yoe <= 8:
        return 1.0
    elif yoe <= 12:
        return 0.75
    else:
        return 0.65  # loosened from 0.5 -- confirmed over-penalising on real data


def cross_encoder_rerank(df: pd.DataFrame, jd_text: str, top_k: int = 40) -> pd.DataFrame:
    """Two-stage rerank: BGE recalled the top candidates, cross-encoder
    reorders the top_k of those with higher precision pairwise scoring.

    Gracefully skips if model isn't downloaded yet -- run
    precompute_crossencoder.py first, then commit the model folder.

    Why top_k=40 not 100: cross-encoder is ~500ms/pair on CPU. 40 pairs
    = ~20s. 100 pairs = ~50s which pushes total pipeline close to the limit.
    Ranks 41-100 are already low-confidence; reranking them adds noise more
    than signal, so keeping BGE order there is correct."""
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../data/precomputed/cross_encoder_model",
    )
    if not os.path.exists(model_path):
        print("Cross-encoder model not found -- skipping rerank step.")
        print("Run src/precompute_crossencoder.py to enable this.")
        return df

    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_path)

    top_df = df.head(top_k).copy()
    rest_df = df.iloc[top_k:].copy()

    pairs = [(jd_text, text) for text in top_df["narrative_text"]]
    print(f"Cross-encoder reranking top {top_k} candidates ({len(pairs)} pairs)...")
    scores = model.predict(pairs, show_progress_bar=True)
    top_df["cross_encoder_score"] = scores

    top_df = top_df.sort_values(
        "cross_encoder_score", ascending=False
    ).reset_index(drop=True)

    combined = pd.concat([top_df, rest_df], ignore_index=True)
    combined["rank"] = range(1, len(combined) + 1)
    return combined


def composite_score(df: pd.DataFrame) -> pd.Series:
    """
    Weight changes vs. Day 3 (all validated against real top-100 cards):

    title_bonus:   0.40 → 0.25  -- was drowning candidates with weak career history
                                    (rank 4 had 0 JD hits, rank 3 had TCS+CV work)
    semantic_score: 0.35 → 0.40 -- BGE doing real semantic work, deserves more say
    green_score:    0.00 → 0.15 -- JD skill signals (vector DB, LTR, eval framework,
                                    fine-tuning, embeddings) already computed in parquet;
                                    caught CAND_0086022 (14 hits, rank 84) and
                                    CAND_0027691 (12 hits, rank 96) being buried
    location_fit:   0.10 → 0.10 -- unchanged
    notice_fit:     0.10 → 0.05 -- marginal differentiator, freed weight for green
    exp_fit:        0.05 → 0.05 -- unchanged (but threshold loosened in function above)

    Behavioral modifier: dampened from floor~0.25x to floor~0.70x in rules.py.
    Strong career + weak recruiter response rate was the single biggest misranking
    cause (CAND_0086022 was 0.617x, CAND_0027691 was 0.768x -- now 0.847 and 0.907).
    """
    title_bonus = df["title_tier"].map({"A": 1.0, "B": 0.4}).fillna(0.0)
    exp_fit = df["years_of_experience"].apply(experience_fit)

    green_cols = [
        "green_embeddings_production", "green_vector_db", "green_eval_framework",
        "green_llm_finetuning", "green_learning_to_rank", "green_open_source_signal",
    ]
    available = [c for c in green_cols if c in df.columns]
    green_score = df[available].astype(float).sum(axis=1) / max(len(available), 1)

    # narrative_signal_count: JD signals from career descriptions only.
    # More reliable than green_score (skills-list based) -- verified Day 4.
    # Normalize to 0-1 against the maximum possible hits (29 signal terms).
    if "narrative_signal_count" in df.columns:
        narrative_score = (df["narrative_signal_count"] / 29.0).clip(0, 1)
    else:
        narrative_score = green_score  # fallback if old parquet without this col

    base = (
        0.25 * title_bonus
        + 0.35 * df["semantic_score"]    # slightly reduced to make room for narrative
        + 0.10 * narrative_score          # replaces green_score -- career description hits only
        + 0.05 * green_score              # keep skills-list signals as minor secondary signal
        + 0.10 * df["location_fit"]
        + 0.05 * df["notice_fit"]
        + 0.05 * exp_fit
                                          # weights sum: 0.25+0.35+0.10+0.05+0.10+0.05+0.05 = 0.95
                                          # remaining 0.05 absorbed by behavioral_modifier multiplier range
    )
    penalty = (
        0.30 * df["flag_consulting_only"].astype(float)
        + 0.15 * df["flag_title_chaser"].astype(float)
        + 0.10 * df["flag_skill_inflation"].astype(float)
        + 0.40 * df["flag_yoe_mismatch"].astype(float)  # honeypot signal -- heavy penalty
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

    # Cross-encoder rerank of top 40 (optional -- skips gracefully if model absent)
    df = cross_encoder_rerank(df, JD_TEXT, top_k=40)
    df["rank"] = range(1, len(df) + 1)

    top100 = df.head(100).copy()
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