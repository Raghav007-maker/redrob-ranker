"""
Day 2: feature extraction + hard filter over the FULL candidate pool.

This is the "100K -> ~5-10K" stage from the architecture diagram. It does two
things in one pass over candidates.jsonl:
  1. A cheap title-relevance filter (is this person even plausibly in scope?)
  2. Pulls in every rules.py flag/score for whoever survives the filter

Output: data/precomputed/features.parquet -- the input to tomorrow's
semantic scoring + fusion stage. Don't commit the parquet file itself
(it's in .gitignore) -- commit this script, anyone can regenerate it.

NOTE on the title filter below: it's a first draft, same as rules.py was.
"engineer" alone would wrongly include Civil/Mechanical/Electrical engineers
(they're ~5% of this pool each) -- so those are explicitly excluded. Expect
to find more false positives/negatives once you look at what passes. That's
expected, not a bug -- tighten it after looking at real output, not before.
"""

import json
import os
import time
import pandas as pd
from rules import hard_red_flags, green_signals, location_fit, notice_period_fit, behavioral_modifier, _narrative_blob

NARRATIVE_JD_SIGNALS = [
    "sentence-transformers", "sentence transformers", "bge", "e5 embedding",
    "openai embeddings",
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "faiss", "vector database", "hybrid search",
    "pgvector", "vector search", "vector store",
    "rag", "retrieval-augmented", "semantic search", "information retrieval",
    "ndcg", "mrr", "map@", "a/b test", "offline-to-online",
    "lora", "qlora", "peft",
    "learning-to-rank", "learning to rank",
]


def narrative_signal_count(candidate: dict) -> int:
    """Count of JD signal terms found in career descriptions only.
    More reliable than the skills-list based green_signals because the
    skills list is heavily salted with random buzzword tags in this dataset.
    This score was verified to correlate better with genuine career-level fit
    during Day 4 hand-validation (candidates with 0 hits in this were
    confirmed non-fits regardless of their skills list)."""
    text = _narrative_blob(candidate)
    return sum(1 for kw in NARRATIVE_JD_SIGNALS if kw in text)

TIER_A_TITLES = {  # ~1,156 candidates, ~1.15% of pool -- the narrow target zone
    "ML Engineer", "AI Research Engineer", "Data Scientist",
    "Senior Software Engineer (ML)", "Computer Vision Engineer",
    "Junior ML Engineer", "AI Specialist", "Recommendation Systems Engineer",
    "Machine Learning Engineer", "Applied ML Engineer", "Search Engineer",
    "AI Engineer", "Senior Data Scientist", "NLP Engineer", "Senior NLP Engineer",
    "Senior Machine Learning Engineer", "Staff Machine Learning Engineer",
    "Senior AI Engineer", "Lead AI Engineer", "Senior Applied Scientist",
}
TIER_B_TITLES = {  # ~30,000 candidates -- plausible adjacent, where "plain
    "Software Engineer", "Full Stack Developer", "Cloud Engineer",        # language" Tier-5 fits per the JD could be hiding
    "Java Developer", ".NET Developer", "DevOps Engineer", "Mobile Developer",
    "Frontend Engineer", "QA Engineer", "Analytics Engineer", "Data Engineer",
    "Data Analyst", "Backend Engineer", "Senior Data Engineer",
    "Senior Software Engineer",
}
# Everything else (12 titles, ~68,000 candidates -- Business Analyst, HR
# Manager, Mechanical Engineer, etc.) is decoy, confirmed against the full
# 47-title vocabulary in candidates.jsonl. Excluded.


def title_tier(title: str) -> str:
    if title in TIER_A_TITLES:
        return "A"
    if title in TIER_B_TITLES:
        return "B"
    return "decoy"


def is_tech_relevant_title(title: str) -> bool:
    return title_tier(title) != "decoy"


def extract_row(candidate: dict) -> dict:
    flags = hard_red_flags(candidate)
    green = green_signals(candidate)
    row = {
        "candidate_id": candidate["candidate_id"],
        "current_title": candidate["profile"]["current_title"],
        "title_tier": title_tier(candidate["profile"]["current_title"]),
        "years_of_experience": candidate["profile"]["years_of_experience"],
        "location": candidate["profile"]["location"],
        "location_fit": location_fit(candidate),
        "notice_fit": notice_period_fit(candidate),
        "behavioral_modifier": behavioral_modifier(candidate),
        "narrative_text": _narrative_blob(candidate),
        "narrative_signal_count": narrative_signal_count(candidate),
    }
    for name, (flag, reason) in flags.items():
        row[f"flag_{name}"] = flag
    for name, val in green.items():
        row[f"green_{name}"] = val
    return row


def main():
    t0 = time.time()
    n_total = 0
    n_passed_filter = 0
    rows = []

    with open(os.environ.get("CANDIDATES_PATH", "../data/raw/candidates.jsonl")) as f:
        for line in f:
            n_total += 1
            c = json.loads(line)
            if not is_tech_relevant_title(c["profile"]["current_title"]):
                continue
            n_passed_filter += 1
            rows.append(extract_row(c))

    df = pd.DataFrame(rows)
    t_filter_and_extract = time.time() - t0

    df.to_parquet("../data/precomputed/features.parquet", index=False)
    t_total = time.time() - t0

    print(f"Total candidates read:        {n_total}")
    print(f"Passed title filter:          {n_passed_filter}  ({100*n_passed_filter/n_total:.1f}%)")
    print(f"Filter + feature extraction:  {t_filter_and_extract:.1f}s")
    print(f"Including parquet write:      {t_total:.1f}s")
    print()
    print("Red flag rates among the candidates who passed the filter:")
    for col in [c for c in df.columns if c.startswith("flag_")]:
        print(f"  {col}: {df[col].sum()} ({100*df[col].mean():.1f}%)")


if __name__ == "__main__":
    main()