"""
JD disqualifiers/preferences from job_description.docx, encoded as rules.

This is a FIRST DRAFT. The JD's hard-coded disqualifiers and preferences were
written in prose by the organizers -- this module turns that prose into
checkable logic over the actual candidate_schema.json fields. Functions
flagged "WEAK PROXY" below mean the schema doesn't directly contain the
field the JD is really asking about (e.g. there's no "has_published_papers"
field) -- so the function approximates it. Validate every WEAK PROXY against
real candidates from sample_candidates.json before you trust it. Day 4 of
the roadmap (hand-checking 30-50 candidates) is specifically for catching
where these are wrong.

Every function returns (bool_or_score, reason_string) so the reason can be
reused directly by the reasoning generator later -- don't regenerate
explanations separately, reuse what these functions already computed.
"""

from datetime import date, datetime

# ---------------------------------------------------------------------------
# Reference lists pulled directly from the JD text
# ---------------------------------------------------------------------------

CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini",
]

EMBEDDING_TERMS = [
    "sentence-transformers", "sentence transformers", "openai embeddings",
    "bge", "e5 embedding", "embedding drift", "index refresh",
    "retrieval-quality", "retrieval quality",
]

VECTOR_DB_TERMS = [
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "faiss", "vector database", "hybrid search",
]

EVAL_FRAMEWORK_TERMS = [
    "ndcg", "mrr", "map@", "mean average precision", "a/b test",
    "ab test", "offline-to-online", "offline to online",
]

LLM_FINETUNE_TERMS = ["lora", "qlora", "peft", "fine-tuning llms", "fine-tuning"]
LTR_TERMS = ["learning-to-rank", "learning to rank", "xgboost ranking", "neural ranking"]

NLP_IR_TERMS = [
    "nlp", "natural language processing", "named entity", "text classification",
    "sentiment analysis", "tokeniz", "transformer model", "language model",
    "text retrieval", "semantic search", "retrieval-augmented", "question answering",
]  # tightened: dropped bare "embedding"/"search"/"ranking" -- too generic,
   # showed up on a Computer Vision Engineer's skill list and suppressed the flag
CV_SPEECH_ROBOTICS_TERMS = [
    "computer vision", "image classification", "speech recognition",
    "tts", "robotics", "gans", "object detection",
]

LEADERSHIP_TITLE_TERMS = [
    "architect", "head of", "vp ", "vice president", "director", "principal",
]

PREFERRED_CITIES = {"pune", "noida"}
TIER1_INDIA_CITIES = {"hyderabad", "mumbai", "delhi", "gurgaon", "gurugram", "bengaluru", "bangalore"}


def _blob(candidate: dict) -> str:
    """All free-text fields, lowercased, concatenated once -- cheap keyword search."""
    parts = [
        candidate["profile"].get("headline", ""),
        candidate["profile"].get("summary", ""),
    ]
    for role in candidate.get("career_history", []):
        parts.append(role.get("title", ""))
        parts.append(role.get("description", ""))
    for s in candidate.get("skills", []):
        parts.append(s.get("name", ""))
    return " ".join(parts).lower()


def consulting_only_career(candidate: dict):
    companies = [c["company"].lower() for c in candidate["career_history"]]
    if not companies:
        return False, ""
    is_consulting = [any(f in comp for f in CONSULTING_FIRMS) for comp in companies]
    if all(is_consulting):
        return True, "Entire career history is at consulting firms (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini) -- explicit JD disqualifier unless prior product-company experience exists."
    return False, ""


def architect_without_recent_code(candidate: dict):
    """WEAK PROXY: schema has no 'last wrote production code' field. Approximated
    via current title containing a leadership term and current tenure > 18mo.
    NOTE: verified against the full 100K pool -- this dataset's title taxonomy
    has zero Architect/Director/VP/Head-of titles, so this never fires here.
    Left in for safety (harmless no-op) in case that's wrong on data you add."""
    current = next((c for c in candidate["career_history"] if c.get("is_current")), None)
    if not current:
        return False, ""
    title = current["title"].lower()
    if any(t in title for t in LEADERSHIP_TITLE_TERMS) and current.get("duration_months", 0) > 18:
        return True, f"Current title '{current['title']}' is leadership/architect, held {current['duration_months']} months -- JD explicitly wants hands-on coders, not 18mo+ in non-coding roles."
    return False, ""


def _narrative_blob(candidate: dict) -> str:
    """Titles + role descriptions only -- NOT the skills list. The skills
    array in this dataset is salted with random unrelated buzzword tags
    (verified: a Data Engineer with 'GANs', 'Sales', 'Figma' all in one
    list) -- it's noise for domain classification. Career narrative text
    is coherent and reflects what someone actually did."""
    parts = [
        candidate["profile"].get("headline", ""),
        candidate["profile"].get("summary", ""),
    ]
    for role in candidate.get("career_history", []):
        parts.append(role.get("title", ""))
        parts.append(role.get("description", ""))
    return " ".join(parts).lower()


def cv_speech_without_nlp(candidate: dict):
    blob = _narrative_blob(candidate)
    has_cv_speech = any(t in blob for t in CV_SPEECH_ROBOTICS_TERMS)
    has_nlp_ir = any(t in blob for t in NLP_IR_TERMS)
    if has_cv_speech and not has_nlp_ir:
        return True, "Career narrative (titles + role descriptions) shows CV/speech/robotics work with no NLP/IR exposure -- explicit JD disqualifier."
    return False, ""


def title_chase_pattern(candidate: dict):
    """2+ company changes with <18mo tenure each, while titles escalate (junior->senior->staff etc)."""
    history = sorted(candidate["career_history"], key=lambda c: c["start_date"])
    short_hops = sum(1 for c in history if c.get("duration_months", 999) < 18 and not c.get("is_current"))
    if short_hops >= 2 and len(history) >= 3:
        return True, f"{short_hops} prior roles held under 18 months each -- matches the JD's 'title-chaser' disqualifier pattern."
    return False, ""


def skill_claim_vs_assessment_gap(candidate: dict):
    """Self-reported proficiency vs Redrob's own tested assessment score.
    Returns the worst (proficiency, assessment) mismatch found, if any."""
    scores = candidate["redrob_signals"].get("skill_assessment_scores", {})
    worst = None
    for s in candidate.get("skills", []):
        name = s["name"]
        prof = s.get("proficiency")
        if name in scores and prof in ("advanced", "expert"):
            assessed = scores[name]
            if assessed < 50:
                gap = (50 - assessed)
                if worst is None or gap > worst[0]:
                    worst = (gap, name, prof, assessed)
    if worst:
        _, name, prof, assessed = worst
        return True, f"Claims '{prof}' in {name} but Redrob's own assessment scored it {assessed}/100 -- claim not backed by tested skill."
    return False, ""


def location_fit(candidate: dict) -> float:
    loc = candidate["profile"].get("location", "").lower()
    country = candidate["profile"].get("country", "")
    relocate = candidate["redrob_signals"].get("willing_to_relocate", False)
    if any(c in loc for c in PREFERRED_CITIES):
        return 1.0
    if any(c in loc for c in TIER1_INDIA_CITIES):
        return 0.8
    if country == "India":
        return 0.6 if relocate else 0.4
    return 0.3 if relocate else 0.1  # outside India: JD says case-by-case, no visa sponsorship


def notice_period_fit(candidate: dict) -> float:
    days = candidate["redrob_signals"].get("notice_period_days", 90)
    if days <= 30:
        return 1.0
    if days <= 60:
        return 0.7
    return 0.4  # still in scope per JD, "but the bar gets higher"


def green_signals(candidate: dict) -> dict:
    blob = _blob(candidate)
    return {
        "embeddings_production": any(t in blob for t in EMBEDDING_TERMS),
        "vector_db": any(t in blob for t in VECTOR_DB_TERMS),
        "eval_framework": any(t in blob for t in EVAL_FRAMEWORK_TERMS),
        "llm_finetuning": any(t in blob for t in LLM_FINETUNE_TERMS),
        "learning_to_rank": any(t in blob for t in LTR_TERMS),
        "open_source_signal": candidate["redrob_signals"].get("github_activity_score", -1) > 30,
    }


def hard_red_flags(candidate: dict) -> dict:
    """Each value is (flag: bool, reason: str). Use these to exclude or heavily
    down-rank -- not necessarily a hard zero, your call on weighting.

    cv_speech_without_nlp is deliberately excluded here. Tried 3 variants
    (skills list, full narrative, narrative-only) -- every one of them
    trips on generic ML buzzword salad that this dataset salts into both
    skills lists AND self-written summaries, independent of a candidate's
    actual specialization. Not a tuning problem -- a property of how this
    data was generated. Re-include only if you find a more reliable signal;
    don't ship a rule you've already disproved 3 times."""
    return {
        "consulting_only": consulting_only_career(candidate),
        "architect_no_code": architect_without_recent_code(candidate),
        "title_chaser": title_chase_pattern(candidate),
        "skill_inflation": skill_claim_vs_assessment_gap(candidate),
    }


if __name__ == "__main__":
    # Quick self-test against the sample bundle -- run this before trusting any of it.
    import json
    with open("sample_candidates.json") as f:
        sample = json.load(f)
    for c in sample[:5]:
        print(c["candidate_id"], c["profile"]["current_title"], "|", c["profile"]["location"])
        for name, (flag, reason) in hard_red_flags(c).items():
            if flag:
                print("  RED FLAG:", name, "-", reason)
        print("  green:", {k: v for k, v in green_signals(c).items() if v})
        print("  location_fit:", location_fit(c), "notice_fit:", notice_period_fit(c))
        print()
