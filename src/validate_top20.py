"""
Day 4: hand-validation helper.

Prints a readable profile card for each of your top N candidates so you
can eyeball them without parsing JSON manually. Run from src/:

    python validate_top20.py           # reviews top 20
    python validate_top20.py --n 50    # reviews top 50
    python validate_top20.py --id CAND_0043860  # single candidate deep-dive

What to look for in each card (this is the whole point of Day 4):
  - Does the career history actually match the claimed title/YOE?
  - Do the role descriptions mention real ML production work, or just buzzwords?
  - Does the BGE semantic fit score feel right for what they actually did?
  - Any red flags the rules missed that you can see by eye?
  - Any Tier B candidates who deserved top-100 but got crowded out by
    the 0.40 title weight? (look at the tail of your top 100 too)
"""

import argparse
import json
import pandas as pd


CONSULTING_FIRMS = ["tcs", "tata consultancy", "infosys", "wipro",
                    "accenture", "cognizant", "capgemini"]

JD_MUST_HAVES = [
    # embedding frameworks
    "sentence-transformers", "sentence transformers", "bge", "e5 embedding",
    "openai embeddings",
    # vector DBs (matches VECTOR_DB_TERMS in rules.py -- keep in sync)
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "faiss", "vector database", "hybrid search",
    "pgvector", "pg vector", "vector search", "vector store",
    # retrieval / RAG
    "rag", "retrieval-augmented", "semantic search", "information retrieval",
    # eval frameworks
    "ndcg", "mrr", "map@", "a/b test", "offline-to-online",
    # fine-tuning
    "lora", "qlora", "peft",
    # ranking
    "learning-to-rank", "learning to rank", "xgboost ranking",
]


def signal_hits(text: str) -> list:
    t = text.lower()
    return [kw for kw in JD_MUST_HAVES if kw in t]


def print_card(rank_row: pd.Series, candidate: dict):
    sep = "─" * 72
    sig = candidate["redrob_signals"]
    profile = candidate["profile"]

    print(f"\n{'═'*72}")
    print(f"  RANK #{int(rank_row['rank'])}  |  {rank_row['candidate_id']}  |  score {rank_row['score']:.4f}")
    print(sep)
    print(f"  Title   : {profile['current_title']}  ({profile['years_of_experience']} yrs)")
    print(f"  Location: {profile.get('location','?')}  |  Notice: {sig.get('notice_period_days','?')}d  |  Open to work: {sig.get('open_to_work_flag','?')}")
    print(f"  Relocate: {sig.get('willing_to_relocate','?')}  |  Last active: {sig.get('last_active_date','?')}")
    print(f"  Response rate: {sig.get('recruiter_response_rate','?'):.0%}  |  Interview completion: {sig.get('interview_completion_rate','?'):.0%}")

    print(sep)
    print("  CAREER HISTORY:")
    history = sorted(candidate.get("career_history", []),
                     key=lambda c: c.get("start_date", ""), reverse=True)
    for role in history:
        current_tag = " ← CURRENT" if role.get("is_current") else ""
        company = role.get("company", "?")
        is_consulting = any(f in company.lower() for f in CONSULTING_FIRMS)
        consulting_tag = " ⚠ CONSULTING" if is_consulting else ""
        print(f"    [{role.get('start_date','?')} – {role.get('end_date','present')}]  "
              f"{role.get('title','?')} @ {company}"
              f"  ({role.get('duration_months','?')} mo){current_tag}{consulting_tag}")
        desc = role.get("description", "")
        if desc:
            hits = signal_hits(desc)
            hit_tag = f"  🟢 JD signals: {hits}" if hits else ""
            print(f"      {desc[:180].strip()}...{hit_tag}")

    print(sep)
    print("  SKILLS (self-reported + assessed):")
    assessed = sig.get("skill_assessment_scores", {})
    skills = candidate.get("skills", [])
    for s in skills:
        name = s.get("name", "?")
        prof = s.get("proficiency", "?")
        score = assessed.get(name)
        score_str = f"  assessed {score:.0f}/100" if score is not None else ""
        mismatch = ""
        if score is not None and prof in ("advanced", "expert") and score < 50:
            mismatch = "  ⚠ CLAIM-SCORE GAP"
        print(f"    {name:<28} {prof:<12}{score_str}{mismatch}")

    print(sep)
    blob = " ".join([
        profile.get("headline", ""),
        profile.get("summary", ""),
        *[r.get("description", "") for r in history],
        *[s.get("name", "") for s in skills],
    ])
    jd_hits = signal_hits(blob)
    print(f"  JD SIGNAL HITS ({len(jd_hits)}): {jd_hits if jd_hits else 'none'}")
    print(f"  GitHub activity score: {sig.get('github_activity_score','?')}")
    print(f"  Profile completeness : {sig.get('profile_completeness_score','?')}")

    print(sep)
    print("  SUMMARY  (profile.summary):")
    print(f"    {profile.get('summary','(none)')[:300]}")
    print()
    print("  YOUR VERDICT:")
    print("    [ ] Rank feels right")
    print("    [ ] Rank too high -- reason: ___________________________")
    print("    [ ] Rank too low  -- reason: ___________________________")
    print("    [ ] Weight to adjust: ___________________________")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Review top N candidates")
    parser.add_argument("--id", type=str, default=None, help="Review a single candidate by ID")
    args = parser.parse_args()

    draft = pd.read_csv("../data/precomputed/submission_draft.csv")

    # Build a lookup index over candidates.jsonl -- one pass, keyed by candidate_id
    print("Building candidate index (one-time, ~5s)...")
    index = {}
    target_ids = set(draft.head(args.n)["candidate_id"]) if not args.id else {args.id}
    with open("../data/raw/candidates.jsonl") as f:
        for line in f:
            c = json.loads(line)
            if c["candidate_id"] in target_ids:
                index[c["candidate_id"]] = c
            if len(index) == len(target_ids):
                break  # found all, stop early
    print(f"Loaded {len(index)} profiles.\n")

    if args.id:
        if args.id not in index:
            print(f"Candidate {args.id} not found in candidates.jsonl")
            return
        row = draft[draft["candidate_id"] == args.id].iloc[0]
        print_card(row, index[args.id])
    else:
        for _, row in draft.head(args.n).iterrows():
            cid = row["candidate_id"]
            if cid in index:
                print_card(row, index[cid])
            else:
                print(f"\n[WARN] {cid} not found in candidates.jsonl -- skipped")

    print("\n" + "═"*72)
    print("  DONE. Tally your verdicts, then adjust weights in rank.py:")
    print("  composite_score() -- title_bonus, semantic_score, exp_fit weights")
    print("  Also check: are any Tier B candidates near rank 100 that should")
    print("  have ranked higher? Run with --n 100 to see the full picture.")
    print("═"*72)


if __name__ == "__main__":
    main()
