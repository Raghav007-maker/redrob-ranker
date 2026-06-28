import pandas as pd

PATCHES = {
    "CAND_0002025": (
        "Senior AI Engineer at Apple (5.9yr); LoRA/QLoRA fine-tuning at Aganitha "
        "and A/B-tested production recommendation at Apple; ranked #2 on BGE semantic "
        "fit — career descriptions lighter on explicit retrieval terms but strong "
        "semantic alignment with JD; 30-day notice."
    ),

    "CAND_0099806": (
        "AI Engineer at Mad Street Den (4.6yr) with hands-on ranking model delivery "
        "(XGBoost/LightGBM, A/B tested in production); BGE semantic score elevated "
        "on strong vector DB skill coverage and ranking background; "
        "30-day notice, active GitHub (86.9)."
    ),

    "CAND_0054546": (
        "AI Research Engineer at Razorpay (4.9yr); 0 JD signals in role descriptions "
        "— career is NLP classification for internal analytics, not production "
        "search/ranking; ranked here on BGE semantic score and Pune location; "
        "borderline fit for this specific JD."
    ),

    "CAND_0008295": (
        "AI Research Engineer at Razorpay (6.5yr); 0 JD signals in role descriptions "
        "— NLP for an internal analytics dashboard, prior role at TCS doing time-series; "
        "limited direct retrieval/ranking evidence; ranked here on BGE semantic fit "
        "and Pune location despite weak career alignment."
    ),

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
            df.loc[mask, "reasoning"] = new_reasoning
            rank = int(df.loc[mask, "rank"].iloc[0])
            print(f"Patched rank #{rank} ({cid})")
            patched += 1
        else:
            print(f"WARNING: {cid} not found in CSV -- skipped")

    df = df.sort_values(["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    dupes = df["reasoning"].duplicated().sum()
    scores = df["score"].tolist()
    violations = sum(
        1 for i in range(1, len(scores)) if scores[i] > scores[i-1] + 1e-6
    )
    print(f"\nPatched {patched} rows")
    print(f"Duplicate reasonings: {dupes}")
    print(f"Score monotonicity violations: {violations}")

    df[["candidate_id", "rank", "score", "reasoning"]].to_csv(path, index=False)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()