"""
Redrob Hackathon — Streamlit Sandbox Demo
==========================================
Upload up to 100 candidate JSON records and get a ranked CSV output.
This satisfies the "sandbox link" requirement in submission_spec Section 10.5.

Deploy free at: https://streamlit.io/cloud
"""

import csv
import io
import json
import streamlit as st
import pandas as pd

# Import our ranker functions
from rank import compute_final_score, build_reasoning

st.set_page_config(page_title="Redrob Candidate Ranker", page_icon="🎯", layout="wide")

st.title("🎯 Redrob Intelligent Candidate Ranker")
st.caption("India Runs Hackathon — Track 1: Data & AI Challenge")

st.markdown("""
This tool ranks candidates against the **Senior AI Engineer — Founding Team** job description 
using a multi-signal scoring system that evaluates:
- **Skill match** (35%) — embeddings, retrieval, vector DBs, Python, NLP
- **Experience fit** (20%) — ideal range 6-8 years
- **Title relevance** (15%) — ML/AI engineer titles score high
- **Behavioral signals** (20%) — activity, response rate, availability
- **Location match** (10%) — Pune/Noida/Delhi NCR preferred
""")

st.divider()

# Upload section
st.subheader("📂 Upload Candidates")
uploaded = st.file_uploader(
    "Upload a JSONL file (one candidate JSON per line, max 100 candidates for demo)",
    type=["jsonl", "json"],
    help="Upload sample_candidates.json or any subset of candidates.jsonl"
)

# Or use sample
use_sample = st.checkbox("Use built-in sample (paste JSON below)", value=False)

sample_json = ""
if use_sample:
    sample_json = st.text_area(
        "Paste candidate JSON records (one per line):",
        height=200,
        placeholder='{"candidate_id": "CAND_0000001", "profile": {...}, ...}'
    )

if st.button("🚀 Run Ranker", type="primary"):
    candidates = []

    if uploaded:
        content = uploaded.read().decode("utf-8")
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                cand = json.loads(line)
                candidates.append(cand)
            except json.JSONDecodeError:
                st.warning(f"Skipped malformed line")
        # Handle pretty-printed single JSON array too
        if not candidates:
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    candidates = data
                elif isinstance(data, dict):
                    candidates = [data]
            except Exception:
                pass

    elif use_sample and sample_json.strip():
        for line in sample_json.strip().split("\n"):
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except Exception:
                    pass

    if not candidates:
        st.error("No valid candidates found. Please upload a JSONL file or paste JSON.")
        st.stop()

    # Cap at 100 for the demo
    if len(candidates) > 100:
        st.info(f"Demo capped at 100 candidates (you uploaded {len(candidates)}). Full run handles 100K.")
        candidates = candidates[:100]

    st.info(f"Scoring {len(candidates)} candidates...")

    # Score
    scored = []
    progress = st.progress(0)
    for i, cand in enumerate(candidates):
        score = compute_final_score(cand)
        scored.append((cand, score))
        progress.progress((i + 1) / len(candidates))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:min(100, len(scored))]

    # Build output dataframe
    rows = []
    for rank, (cand, score) in enumerate(top, start=1):
        profile = cand.get("profile", {})
        signals = cand.get("redrob_signals", {})
        rows.append({
            "rank": rank,
            "candidate_id": cand.get("candidate_id", ""),
            "score": round(score, 4),
            "title": profile.get("current_title", ""),
            "yoe": profile.get("years_of_experience", 0),
            "location": profile.get("location", ""),
            "open_to_work": signals.get("open_to_work_flag", False),
            "response_rate": round(signals.get("recruiter_response_rate", 0), 2),
            "reasoning": build_reasoning(cand, score, rank),
        })

    df = pd.DataFrame(rows)

    st.success(f"✅ Ranked {len(rows)} candidates successfully!")
    st.subheader("📊 Top Candidates")
    st.dataframe(df, use_container_width=True)

    # Download CSV
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for row in rows:
        writer.writerow([row["candidate_id"], row["rank"], row["score"], row["reasoning"]])

    st.download_button(
        "⬇️ Download submission.csv",
        data=csv_buffer.getvalue(),
        file_name="submission.csv",
        mime="text/csv",
    )

    # Score breakdown
    st.subheader("📈 Score Distribution")
    score_df = pd.DataFrame({"score": [s for _, s in scored]})
    st.bar_chart(score_df["score"].value_counts(bins=20).sort_index())

st.divider()
st.caption("Built for India Runs Hackathon by Redrob AI · Track 1: Data & AI Challenge")
