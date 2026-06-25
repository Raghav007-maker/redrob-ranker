"""
redrob-ranker Streamlit sandbox.

Submission spec 10.5: sandbox is mandatory. This app accepts a small
candidate sample (<=100 candidates), runs the full ranking pipeline,
and produces a ranked CSV for download.

Uses TF-IDF semantic scoring (no BGE model load) for fast demo.
Production submission uses precomputed BGE embeddings (baked into Docker).
"""

import json
import sys
import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Add src/ to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from jd_text import JD_TEXT
from rules import (
    hard_red_flags, green_signals, location_fit,
    notice_period_fit, behavioral_modifier, _narrative_blob,
)
from feature_extraction import title_tier
from rank import experience_fit
from generate_reasoning import build_reasoning

st.set_page_config(
    page_title="redrob-ranker | Recruiter Hub",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main Dashboard Header */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .main-header h1 {
        color: white !important;
        font-weight: 700;
        font-size: 2.5rem;
        margin: 0;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    
    /* Glassmorphic card styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.05);
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 75%;
        font-weight: 600;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 10rem;
        margin-right: 5px;
        margin-bottom: 5px;
    }
    
    .badge-tier-a { background-color: #e3f2fd; color: #0d47a1; border: 1px solid #90caf9; }
    .badge-tier-b { background-color: #efebe9; color: #4e342e; border: 1px solid #bcaaa4; }
    .badge-flag-active { background-color: #ffebee; color: #c62828; border: 1px solid #ef9a9a; }
    .badge-flag-inactive { background-color: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7; }
    .badge-green { background-color: #e0f2f1; color: #004d40; border: 1px solid #80cbc4; }
    
    /* Candidate Info Panel */
    .profile-section {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border-left: 5px solid #2a5298;
    }
    
    .profile-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1e3c72;
    }
    
    .reasoning-box {
        background: #f0f4f8;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #1e3c72;
        font-style: italic;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Dashboard Title
st.markdown("""
<div class="main-header">
    <h1>🎯 redrob-ranker | AI Recruiter Hub</h1>
    <p>Premium Candidate Alignment & Ranking Dashboard (Senior AI Engineer)</p>
</div>
""", unsafe_allow_html=True)


def load_candidates(source):
    candidates = []
    if isinstance(source, str):
        with open(source) as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
    else:
        content = source.read().decode("utf-8")
        for line in content.splitlines():
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def run_pipeline(candidates, weights, penalties):
    def extract_row(c):
        flags = hard_red_flags(c)
        green = green_signals(c)
        row = {
            "candidate_id": c["candidate_id"],
            "current_title": c["profile"]["current_title"],
            "title_tier": title_tier(c["profile"]["current_title"]),
            "years_of_experience": c["profile"]["years_of_experience"],
            "location": c["profile"].get("location", ""),
            "location_fit": location_fit(c),
            "notice_fit": notice_period_fit(c),
            "behavioral_modifier": behavioral_modifier(c),
            "narrative_text": _narrative_blob(c),
        }
        for name, (flag, reason) in flags.items():
            row[f"flag_{name}"] = flag
            row[f"flag_{name}_reason"] = reason
        for name, val in green.items():
            row[f"green_{name}"] = val
        return row

    rows = [extract_row(c) for c in candidates]
    df = pd.DataFrame(rows)

    # TF-IDF semantic scoring (fast demo -- production uses BGE)
    corpus = list(df["narrative_text"]) + [JD_TEXT]
    vec = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    matrix = vec.fit_transform(corpus)
    raw = cosine_similarity(matrix[:-1], matrix[-1]).flatten()
    df["semantic_score"] = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

    title_bonus = df["title_tier"].map({"A": 1.0, "B": 0.4}).fillna(0.0)

    green_cols = [
        "green_embeddings_production", "green_vector_db", "green_eval_framework",
        "green_llm_finetuning", "green_learning_to_rank", "green_open_source_signal",
    ]
    available = [c for c in green_cols if c in df.columns]
    green_score = df[available].astype(float).sum(axis=1) / max(len(available), 1)

    exp_fit = df["years_of_experience"].apply(experience_fit)

    # Weighted calculation
    base = (
        weights["title_bonus"] * title_bonus
        + weights["semantic_score"] * df["semantic_score"]
        + weights["green_score"] * green_score
        + weights["location_fit"] * df["location_fit"]
        + weights["notice_fit"] * df["notice_fit"]
        + weights["exp_fit"] * exp_fit
    )
    
    # Penalties
    penalty = (
        penalties["consulting_only"] * df["flag_consulting_only"].astype(float)
        + penalties["title_chaser"] * df["flag_title_chaser"].astype(float)
        + penalties["skill_inflation"] * df["flag_skill_inflation"].astype(float)
    )
    
    df["score"] = (base - penalty).clip(lower=0) * df["behavioral_modifier"]
    
    # Generate human-readable score breakdown
    df["score_breakdown"] = (
        "Title: " + title_bonus.round(2).astype(str)
        + " | Sem: " + df["semantic_score"].round(2).astype(str)
        + " | Green: " + green_score.round(2).astype(str)
        + " | Behav: " + df["behavioral_modifier"].round(2).astype(str)
    )
    
    df = df.sort_values(["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    
    # Generate reasoning for top candidates
    candidate_index = {c["candidate_id"]: c for c in candidates}
    reasonings = []
    for idx, row in df.iterrows():
        c = candidate_index[row["candidate_id"]]
        reasonings.append(build_reasoning(idx + 1, c, row))
    df["reasoning"] = reasonings
    
    return df


# --- SIDEBAR: Controls & Weights ---
with st.sidebar:
    st.header("🛠️ Controls & Parameters")
    
    use_sample = st.checkbox("Use pre-loaded sample (50 candidates)", value=True)
    uploaded = None
    if not use_sample:
        uploaded = st.file_uploader(
            "Upload candidates (.jsonl or .json)",
            type=["jsonl", "json"],
            help="Each line should be a candidate JSON matching the competition schema."
        )

    top_n = st.slider("Show top N", min_value=5, max_value=100, value=20, step=5)
    
    # Weight Tuning Explainer
    with st.expander("⚖️ Adjust Weight Criteria", expanded=False):
        st.caption("Customize the composite ranking formula weights (must sum to 1.0 ideally).")
        w_title = st.slider("Title Tier Match", 0.0, 1.0, 0.25, 0.05)
        w_semantic = st.slider("Semantic JD similarity", 0.0, 1.0, 0.40, 0.05)
        w_green = st.slider("JD Green Signals Match", 0.0, 1.0, 0.15, 0.05)
        w_loc = st.slider("Location Preference", 0.0, 1.0, 0.10, 0.05)
        w_notice = st.slider("Notice Period Match", 0.0, 1.0, 0.05, 0.05)
        w_exp = st.slider("Experience fit (3-8 yrs)", 0.0, 1.0, 0.05, 0.05)
        
        st.divider()
        st.caption("Penalties (subtracted from baseline score):")
        p_consulting = st.slider("Consulting Only Career", 0.0, 0.5, 0.30, 0.05)
        p_chaser = st.slider("Title Chaser Pattern", 0.0, 0.5, 0.15, 0.05)
        p_inflation = st.slider("Skill Assessment Gap", 0.0, 0.5, 0.10, 0.05)

    weights = {
        "title_bonus": w_title,
        "semantic_score": w_semantic,
        "green_score": w_green,
        "location_fit": w_loc,
        "notice_fit": w_notice,
        "exp_fit": w_exp,
    }
    
    penalties = {
        "consulting_only": p_consulting,
        "title_chaser": p_chaser,
        "skill_inflation": p_inflation,
    }

    run_btn = st.button("▶ Run Pipeline", type="primary", use_container_width=True)

# Load Candidates
candidates = []
try:
    if use_sample:
        sample_path = os.path.join(os.path.dirname(__file__), "src", "sample_candidates.json")
        if os.path.exists(sample_path):
            with open(sample_path) as f:
                candidates = json.load(f)
        else:
            st.error(f"Sample file not found: {sample_path}")
            st.stop()
    else:
        if uploaded is not None:
            candidates = load_candidates(uploaded)
except Exception as e:
    st.error(f"Error loading candidates: {e}")

# Process and Rank if candidates exist
if len(candidates) > 0:
    df = run_pipeline(candidates, weights, penalties)
    candidate_index = {c["candidate_id"]: c for c in candidates}
    
    # Overview Metrics Row
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Total Candidates", len(df))
    with col_m2:
        tier_a_count = len(df[df["title_tier"] == "A"])
        st.metric("Tier-A Role Match", f"{tier_a_count} ({int(tier_a_count/len(df)*100)}%)")
    with col_m3:
        clean_profiles = len(df[~(df["flag_consulting_only"] | df["flag_title_chaser"] | df["flag_skill_inflation"])])
        st.metric("Flags-Free Profiles", f"{clean_profiles} ({int(clean_profiles/len(df)*100)}%)")
    with col_m4:
        avg_yoe = df["years_of_experience"].mean()
        st.metric("Avg Experience", f"{avg_yoe:.1f} Yrs")

    st.write("")
    
    # Interactive Filtering UI
    with st.expander("🔍 Search & Advanced Filtering Options", expanded=False):
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            tier_filter = st.multiselect("Filter by Tier", ["A", "B"], default=["A", "B"])
        with col_f2:
            all_locations = sorted(list(df["location"].unique()))
            loc_filter = st.multiselect("Filter by Location", all_locations, default=all_locations)
        with col_f3:
            search_query = st.text_input("Search current title or Candidate ID")

        # Apply Filters
        filtered_df = df[
            df["title_tier"].isin(tier_filter) &
            df["location"].isin(loc_filter)
        ]
        if search_query:
            q = search_query.lower()
            filtered_df = filtered_df[
                filtered_df["candidate_id"].str.lower().str.contains(q) | 
                filtered_df["current_title"].str.lower().str.contains(q)
            ]
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Ranked Candidates", "👤 Candidate Detail Explorer", "📊 Insights & Distribution", "📄 Fixed Job Description"])

    with tab1:
        st.subheader("Candidate Rankings List")
        
        # Display table columns
        display_cols = ["rank", "candidate_id", "current_title", "title_tier", "years_of_experience",
                        "location", "score", "score_breakdown"]
        
        st.dataframe(
            filtered_df[display_cols].head(top_n),
            use_container_width=True,
            hide_index=True,
        )

        col_d1, col_d2 = st.columns([1, 4])
        with col_d1:
            out = filtered_df[["candidate_id", "rank", "score", "reasoning"]].head(top_n).copy()
            out["score"] = out["score"].round(4)
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download Ranked CSV",
                data=csv_bytes,
                file_name="redrob_ranked.csv",
                mime="text/csv",
                use_container_width=True
            )

    with tab2:
        st.subheader("Deep Dive Candidate Profile")
        
        # Candidate Selector dropdown
        cand_list = filtered_df.head(top_n)["candidate_id"].tolist()
        if not cand_list:
            st.info("No candidates match your current filter settings.")
        else:
            selected_id = st.selectbox("Select Candidate to Inspect", cand_list)
            
            c = candidate_index[selected_id]
            row_data = filtered_df[filtered_df["candidate_id"] == selected_id].iloc[0]
            
            # Layout columns for candidate card
            c_col1, c_col2 = st.columns([2, 3])
            
            with c_col1:
                st.markdown(f"### Profile Summary")
                st.markdown(f"**Name:** {c['profile']['anonymized_name']}")
                st.markdown(f"**Current Title:** {c['profile']['current_title']}")
                st.markdown(f"**Company:** {c['profile'].get('current_company', 'N/A')} (Size: {c['profile'].get('current_company_size', 'N/A')})")
                st.markdown(f"**Years of Experience:** {c['profile']['years_of_experience']} Years")
                st.markdown(f"**Location:** {c['profile'].get('location', '')}, {c['profile'].get('country', '')}")
                
                # Render Badges
                st.write("")
                st.markdown("**Taxonomy Tier:**")
                if row_data['title_tier'] == 'A':
                    st.markdown('<span class="badge badge-tier-a">Tier-A Title Match</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge badge-tier-b">Tier-B Title Match</span>', unsafe_allow_html=True)
                
                # Active Red Flags
                st.markdown("**Hard Flag Checks:**")
                has_red = False
                for flag_name in ["consulting_only", "architect_no_code", "title_chaser", "skill_inflation"]:
                    is_active = row_data[f"flag_{flag_name}"]
                    if is_active:
                        has_red = True
                        st.markdown(f'<span class="badge badge-flag-active">⚠️ {flag_name.replace("_", " ").title()}</span>', unsafe_allow_html=True)
                        st.caption(f"Reason: *{row_data[f'flag_{flag_name}_reason']}*")
                if not has_red:
                    st.markdown('<span class="badge badge-flag-inactive">✅ No Red Flags Detected</span>', unsafe_allow_html=True)

                # Notice period and reloc
                st.write("")
                st.markdown("**Availability & Engagement:**")
                notice_days = c["redrob_signals"].get("notice_period_days", 90)
                st.write(f"- **Notice Period:** {notice_days} days")
                st.write(f"- **Recruiter Response Rate:** {int(c['redrob_signals'].get('recruiter_response_rate', 1.0)*100)}%")
                st.write(f"- **Interview Completion Rate:** {int(c['redrob_signals'].get('interview_completion_rate', 1.0)*100)}%")
                st.write(f"- **Github Activity Score:** {c['redrob_signals'].get('github_activity_score', -1)}")

            with c_col2:
                st.markdown(f"### Match Explanation")
                st.markdown(f'<div class="reasoning-box">{row_data["reasoning"]}</div>', unsafe_allow_html=True)
                
                st.markdown("### Skill Assessment Profile")
                scores = c["redrob_signals"].get("skill_assessment_scores", {})
                skills_list = c.get("skills", [])
                
                if not skills_list:
                    st.caption("No reported skills available.")
                else:
                    skills_df_rows = []
                    for s in skills_list:
                        name = s["name"]
                        prof = s.get("proficiency", "unknown")
                        test_score = scores.get(name, "Not Tested")
                        skills_df_rows.append({
                            "Skill Name": name,
                            "Self-Reported Proficiency": prof,
                            "Redrob Tested Assessment Score": test_score
                        })
                    st.table(pd.DataFrame(skills_df_rows))

                # Green Signals
                st.markdown("### Detected JD Green Signals")
                green_signals_found = []
                for sig_name in ["embeddings_production", "vector_db", "eval_framework", "llm_finetuning", "learning_to_rank", "open_source_signal"]:
                    if row_data[f"green_{sig_name}"]:
                        green_signals_found.append(sig_name.replace("_", " ").title())
                if green_signals_found:
                    for g in green_signals_found:
                        st.markdown(f'<span class="badge badge-green">✔ {g}</span>', unsafe_allow_html=True)
                else:
                    st.caption("No specific advanced keywords matched.")

            st.write("")
            st.markdown("### Career History Timeline")
            for job in c.get("career_history", []):
                with st.container():
                    st.markdown(f"**{job['title']}** at **{job['company']}**")
                    st.caption(f"{job['start_date']} to {job['end_date'] or 'Present'} ({job.get('duration_months', 0)} months)")
                    st.write(job.get("description", ""))
                    st.write("---")

    with tab3:
        st.subheader("Score Distribution Analysis")
        
        # Interactive distribution chart
        st.bar_chart(filtered_df.head(top_n).set_index("candidate_id")["score"])
        
        # Key statistics table
        st.subheader("Summary Statistics")
        desc = filtered_df[["score", "years_of_experience", "behavioral_modifier", "semantic_score"]].describe()
        st.dataframe(desc, use_container_width=True)

    with tab4:
        st.subheader("Senior AI Engineer Job Description")
        st.text_area("JD Text", JD_TEXT.strip(), height=400, disabled=True)

else:
    st.info("Upload candidate JSONL/JSON data or check 'Use pre-loaded sample' in the sidebar to start ranking.")
