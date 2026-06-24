"""
Patches the five Stage 4 risk entries in submission_final.csv.
Run once from src/ after generate_reasoning.py:

    python patch_reasoning.py

All replacement text is grounded in actual candidate data
(verified against the Day 4 profile card output).
"""

import pandas as pd

PATCHES = {
    # Rank 2 -- CAND_0002025, Senior AI Engineer @ Apple (5.9yr)
    # Career: A/B tested production rec system at Apple + LoRA/QLoRA at Aganitha
    # Skills list (self-reported, not cited as production): Weaviate, Pinecone,
    # FAISS, Sentence Transformers, QLoRA -- BGE picked these up semantically
    # Fix: remove "Strongest signal in pool" (factually wrong vs rank 1)
    "CAND_0002025": (
        "Senior AI Engineer at Apple (5.9yr); LoRA/QLoRA fine-tuning at Aganitha "
        "and A/B-tested production recommendation at Apple; ranked #2 on BGE semantic "
        "fit — career descriptions lighter on explicit retrieval terms but strong "
        "semantic alignment with JD; 30-day notice."
    ),

    # Rank 10 -- CAND_0099806, AI Engineer @ Mad Street Den (4.6yr)
    # Career: XGBoost/LightGBM ranking models + A/B testing at Mad Street Den
    #         and upGrad -- genuine ranking work, just not named with BGE/FAISS terms
    # Skills: Sentence Transformers, FAISS, Weaviate, Qdrant, Elasticsearch, pgvector
    # Fix: explain why top-10 despite few explicit signals
    "CAND_0099806": (
        "AI Engineer at Mad Street Den (4.6yr) with hands-on ranking model delivery "
        "(XGBoost/LightGBM, A/B tested in production); BGE semantic score elevated "
        "on strong vector DB skill coverage and ranking background; "
        "30-day notice, active GitHub (86.9)."
    ),

    # Rank 14 -- CAND_0054546, AI Research Engineer @ Razorpay (4.9yr)
    # Career: NLP pipelines for internal dashboard -- NOT search/ranking
    # 0 signals in career descriptions; BGE pulled it high on generic ML text
    # Fix: be honest that this is a BGE-elevated borderline case
    "CAND_0054546": (
        "AI Research Engineer at Razorpay (4.9yr); 0 JD signals in role descriptions "
        "— career is NLP classification for internal analytics, not production "
        "search/ranking; ranked here on BGE semantic score and Pune location; "
        "borderline fit for this specific JD."
    ),

    # Rank 19 -- CAND_0008295, AI Research Engineer @ Razorpay (6.5yr) + TCS
    # Career: NLP for internal feedback dashboard + TCS time-series
    # 0 signals; BGE elevated on generic ML language
    # Fix: honest about the gap, consistent with rank 19 tone
    "CAND_0008295": (
        "AI Research Engineer at Razorpay (6.5yr); 0 JD signals in role descriptions "
        "— NLP for an internal analytics dashboard, prior role at TCS doing time-series; "
        "limited direct retrieval/ranking evidence; ranked here on BGE semantic fit "
        "and Pune location despite weak career alignment."
    ),

    # Rank 22 -- CAND_0093763, Senior SWE (ML) @ Swiggy (5yr)
    # Career: time-series forecasting + NLP classification -- no search/ranking
    # Fix: cite the real concern (no retrieval work), not just the notice period
    "CAND_0093763": (
        "Senior Software Engineer (ML) at Swiggy (5yr); 0 JD signals in role "
        "descriptions — career spans time-series forecasting and NLP classification, "
        "not retrieval or ranking; included on title match and BGE score; "
        "limited fit for this specific JD; 90-day notice."
    ),
}


def main():
    path = "../data/precomputed/submission_final.csv"
    df = pd.read_csv(path)

    patched = 0
    for cid, new_reasoning in PATCHES.items():
        mask = df["candidate_id"] == cid
        if mask.any():
            old = df.loc[mask, "reasoning"].iloc[0]
            df.loc[mask, "reasoning"] = new_reasoning
            rank = int(df.loc[mask, "rank"].iloc[0])
            print(f"Patched rank #{rank} ({cid})")
            print(f"  OLD: {old}")
            print(f"  NEW: {new_reasoning}")
            print()
            patched += 1
        else:
            print(f"WARNING: {cid} not found in CSV -- skipped")

    # Re-validate
    dupes = df["reasoning"].duplicated().sum()
    scores = df["score"].tolist()
    violations = sum(
        1 for i in range(1, len(scores)) if scores[i] > scores[i-1] + 1e-6
    )
    print(f"Patched {patched} rows")
    print(f"Duplicate reasonings after patch: {dupes}")
    print(f"Score monotonicity violations: {violations}")

    df.to_csv(path, index=False)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
