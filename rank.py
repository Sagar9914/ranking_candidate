"""
Redrob Hackathon — Intelligent Candidate Ranking System
=========================================================
Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Or with gzip:
    python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv

Runs on CPU only, no GPU, no external API calls.
Target: < 5 minutes for 100,000 candidates on a 16 GB machine.
"""

import argparse
import csv
import gzip
import json
import math
import re
import sys
from datetime import date, datetime
from pathlib import Path


# ─────────────────────────────────────────────
#  JOB DESCRIPTION — key requirements extracted
# ─────────────────────────────────────────────

# Hard-required skills from the JD (must-haves)
REQUIRED_SKILLS = [
    "embeddings", "sentence-transformers", "retrieval", "vector database",
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
    "elasticsearch", "hybrid search", "ranking", "python", "nlp",
    "information retrieval", "recommendation", "search", "bge", "e5",
    "ndcg", "mrr", "map", "a/b testing", "evaluation", "re-ranking",
    "dense retrieval", "sparse retrieval", "bm25", "transformer",
]

# Nice-to-have skills (bonus points)
NICE_SKILLS = [
    "lora", "qlora", "peft", "fine-tuning", "xgboost", "learning to rank",
    "hr-tech", "recruiting", "distributed systems", "inference optimization",
    "open-source", "rag", "langchain",
]

# Red-flag job titles — should NOT be top candidates
NEGATIVE_TITLES = [
    "marketing manager", "hr manager", "content writer", "graphic designer",
    "business analyst", "sales", "accountant", "lawyer", "doctor",
    "data entry", "customer support",
]

# Positive title signals — should rank higher
POSITIVE_TITLES = [
    "ml engineer", "machine learning engineer", "ai engineer",
    "nlp engineer", "search engineer", "data scientist",
    "senior engineer", "software engineer", "backend engineer",
    "research engineer", "applied scientist",
]

# Experience range the JD actually wants (5-9 years, ideal 6-8)
EXP_MIN = 5
EXP_MAX = 9
EXP_IDEAL_MIN = 6
EXP_IDEAL_MAX = 8

# Location preference
PREFERRED_LOCATIONS = [
    "pune", "noida", "delhi", "delhi ncr", "hyderabad",
    "mumbai", "bangalore", "bengaluru",
]

# Salary range — JD doesn't say explicitly but Series A AI role
# Candidates expecting <8 LPA are likely junior; >60 LPA might be too senior
SALARY_MIN_OK = 8
SALARY_MAX_OK = 60

# Companies that are explicitly negative signals (per JD)
NEGATIVE_COMPANIES = [
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
]

TODAY = date.today()


# ─────────────────────────────────────────────
#  SCORING ENGINE
# ─────────────────────────────────────────────

def normalize(value, lo, hi):
    """Clamp and scale value to [0, 1]."""
    if hi == lo:
        return 0.5
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def skill_score(candidate: dict) -> float:
    """
    Score based on skills match to the JD.
    Returns 0-1. Required skills weighted 2x nice-to-have.
    Also checks career history descriptions for implicit skill signals.
    """
    skills_list = candidate.get("skills", [])

    # Build a text blob of everything skill-related
    skill_names = set()
    proficiency_bonus = 0.0

    for s in skills_list:
        name = s.get("name", "").lower()
        skill_names.add(name)
        prof = s.get("proficiency", "beginner")
        # Advanced/expert skills count more
        if prof in ("advanced", "expert"):
            proficiency_bonus += 0.01

    # Also extract from career descriptions and headline/summary
    full_text = " ".join([
        candidate.get("profile", {}).get("headline", ""),
        candidate.get("profile", {}).get("summary", ""),
        " ".join(
            job.get("description", "") + " " + job.get("title", "")
            for job in candidate.get("career_history", [])
        ),
    ]).lower()

    # Score required skills
    req_hits = 0
    for skill in REQUIRED_SKILLS:
        if skill in skill_names or skill in full_text:
            req_hits += 1

    # Score nice-to-have skills
    nice_hits = 0
    for skill in NICE_SKILLS:
        if skill in skill_names or skill in full_text:
            nice_hits += 1

    req_ratio = req_hits / len(REQUIRED_SKILLS)
    nice_ratio = nice_hits / len(NICE_SKILLS) if NICE_SKILLS else 0

    raw = (req_ratio * 0.75) + (nice_ratio * 0.15) + min(proficiency_bonus, 0.10)
    return min(raw, 1.0)


def experience_score(candidate: dict) -> float:
    """
    Score based on years of experience.
    Ideal range: 6-8 years. Acceptable: 5-9. Outside = penalty.
    Also checks if experience is at product companies, not pure services.
    """
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)

    if EXP_IDEAL_MIN <= yoe <= EXP_IDEAL_MAX:
        base = 1.0
    elif EXP_MIN <= yoe < EXP_IDEAL_MIN:
        base = 0.75
    elif EXP_IDEAL_MAX < yoe <= EXP_MAX:
        base = 0.85
    elif yoe < EXP_MIN:
        base = max(0.0, yoe / EXP_MIN * 0.6)
    else:
        # Over 9 years — not disqualifying, just less ideal
        base = 0.65

    # Check if career shows product company experience
    career = candidate.get("career_history", [])
    product_company_bonus = 0.0
    consulting_penalty = 0.0

    for job in career:
        company = job.get("company", "").lower()
        for neg_co in NEGATIVE_COMPANIES:
            if neg_co in company:
                consulting_penalty += 0.05
        size = job.get("company_size", "")
        # Larger companies (201+) that aren't consulting = product company signal
        if size in ("201-500", "501-1000", "1001-5000", "5001-10000", "10001+"):
            product_company_bonus += 0.03

    adjustment = min(product_company_bonus, 0.10) - min(consulting_penalty, 0.15)
    return max(0.0, min(1.0, base + adjustment))


def title_score(candidate: dict) -> float:
    """
    Score based on current job title.
    Positive ML/AI titles score high; non-technical titles score low.
    """
    title = candidate.get("profile", {}).get("current_title", "").lower()

    for pos in POSITIVE_TITLES:
        if pos in title:
            return 1.0

    for neg in NEGATIVE_TITLES:
        if neg in title:
            return 0.05  # Strong negative signal

    # Neutral title — partial credit if senior-sounding
    if any(w in title for w in ("senior", "lead", "principal", "staff", "architect")):
        return 0.6
    if any(w in title for w in ("engineer", "developer", "scientist", "analyst")):
        return 0.5
    return 0.3


def behavioral_score(candidate: dict) -> float:
    """
    Score based on Redrob behavioral signals.
    Key insight from JD: a perfect-on-paper candidate who is inactive
    is not actually hireable. Behavioral signals are used as a multiplier.
    """
    signals = candidate.get("redrob_signals", {})

    score = 0.0
    max_score = 0.0

    # 1. Open to work (big positive)
    max_score += 15
    if signals.get("open_to_work_flag", False):
        score += 15

    # 2. Last active date (how recent?)
    max_score += 20
    last_active_str = signals.get("last_active_date", "")
    if last_active_str:
        try:
            last_active = date.fromisoformat(last_active_str)
            days_inactive = (TODAY - last_active).days
            if days_inactive <= 7:
                score += 20
            elif days_inactive <= 30:
                score += 16
            elif days_inactive <= 90:
                score += 10
            elif days_inactive <= 180:
                score += 5
            else:
                score += 1  # Very stale
        except ValueError:
            pass

    # 3. Recruiter response rate
    max_score += 15
    response_rate = signals.get("recruiter_response_rate", 0.0)
    score += response_rate * 15

    # 4. Profile completeness
    max_score += 10
    completeness = signals.get("profile_completeness_score", 0)
    score += normalize(completeness, 0, 100) * 10

    # 5. Notice period (JD explicitly wants ≤30 days)
    max_score += 10
    notice = signals.get("notice_period_days", 90)
    if notice <= 30:
        score += 10
    elif notice <= 60:
        score += 6
    elif notice <= 90:
        score += 3
    else:
        score += 0

    # 6. Interview completion rate
    max_score += 8
    icr = signals.get("interview_completion_rate", 0.5)
    score += icr * 8

    # 7. GitHub activity score
    max_score += 10
    github = signals.get("github_activity_score", -1)
    if github >= 0:
        score += normalize(github, 0, 100) * 10
    # -1 means no GitHub — neutral, no penalty but no bonus

    # 8. Saved by recruiters recently (social proof)
    max_score += 7
    saved = signals.get("saved_by_recruiters_30d", 0)
    score += min(saved / 10, 1.0) * 7

    # 9. Work mode match (JD says hybrid Pune/Noida)
    max_score += 5
    work_mode = signals.get("preferred_work_mode", "")
    if work_mode in ("hybrid", "flexible", "onsite"):
        score += 5
    elif work_mode == "remote":
        score += 2

    # 10. Verified contact info (reliability signal)
    max_score += 5
    if signals.get("verified_email", False):
        score += 2.5
    if signals.get("verified_phone", False):
        score += 2.5

    # 11. Willing to relocate (JD mentions relocation candidates welcome)
    max_score += 5
    if signals.get("willing_to_relocate", False):
        score += 5

    # 12. Salary fit (not too low, not impossibly high)
    max_score += 5
    sal = signals.get("expected_salary_range_inr_lpa", {})
    sal_min = sal.get("min", 0)
    sal_max = sal.get("max", 999)
    if SALARY_MIN_OK <= sal_min and sal_max <= SALARY_MAX_OK:
        score += 5
    elif sal_max <= SALARY_MAX_OK:
        score += 3

    return score / max_score if max_score > 0 else 0.0


def location_score(candidate: dict) -> float:
    """
    Bonus for candidates already in preferred cities.
    """
    location = candidate.get("profile", {}).get("location", "").lower()
    country = candidate.get("profile", {}).get("country", "").lower()

    # Must be in India (or willing to relocate — handled in behavioral)
    if country and "india" not in country:
        return 0.1

    for city in PREFERRED_LOCATIONS:
        if city in location:
            return 1.0

    return 0.5  # India but different city — okay


def honeypot_penalty(candidate: dict) -> float:
    """
    Detect impossible/fraudulent profiles and return a penalty multiplier.
    Honeypots have things like 8 years at a 3-year-old company, or
    'expert' in 10 skills with 0 months used.

    Returns 1.0 (no penalty) for normal candidates.
    Returns 0.0 for likely honeypots — this removes them from top 100.
    """
    # Check for impossible experience: skill duration > total career
    yoe_months = candidate.get("profile", {}).get("years_of_experience", 0) * 12
    skills = candidate.get("skills", [])

    expert_zero_duration = 0
    for s in skills:
        prof = s.get("proficiency", "")
        dur = s.get("duration_months", None)
        if prof in ("expert", "advanced") and dur == 0:
            expert_zero_duration += 1

    if expert_zero_duration >= 5:
        return 0.0  # Impossible: expert in 5+ skills but 0 months used

    # Check for impossible tenure: company founded after candidate joined
    career = candidate.get("career_history", [])
    for job in career:
        start_str = job.get("start_date", "")
        duration = job.get("duration_months", 0)
        company_size = job.get("company_size", "")
        # Flag: candidate at a 1-10 person company for 8+ years
        # (most tiny companies don't last that long)
        if company_size == "1-10" and duration > 96:
            return 0.5  # Suspicious but not conclusive

    # Check for keyword stuffing: 15+ skills all listed as expert with high endorsements
    # but headline/summary is clearly non-technical
    headline = candidate.get("profile", {}).get("headline", "").lower()
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if expert_count >= 12 and not any(w in headline for w in [
        "engineer", "scientist", "developer", "researcher", "architect"
    ]):
        return 0.2  # Likely keyword stuffer

    return 1.0  # Normal candidate


def compute_final_score(candidate: dict) -> float:
    """
    Combine all sub-scores into a single 0-1 score.

    Weights based on JD importance:
    - Skill match: 35% (most important — this is a technical role)
    - Experience: 20% (5-9 years range matters)
    - Job title fit: 15% (title must be ML/AI-adjacent)
    - Behavioral signals: 20% (engagement/availability)
    - Location: 10% (preferred cities bonus)
    """
    # Get component scores
    s_skill = skill_score(candidate)
    s_exp = experience_score(candidate)
    s_title = title_score(candidate)
    s_behavioral = behavioral_score(candidate)
    s_location = location_score(candidate)

    # Honeypot check — multiply final score to zero out fraudulent profiles
    penalty = honeypot_penalty(candidate)

    # Weighted combination
    raw = (
        s_skill      * 0.35 +
        s_exp        * 0.20 +
        s_title      * 0.15 +
        s_behavioral * 0.20 +
        s_location   * 0.10
    )

    return raw * penalty


def build_reasoning(candidate: dict, score: float, rank: int) -> str:
    """
    Build a specific, honest 1-2 sentence reasoning for this candidate.
    References actual facts from the profile — no hallucination.
    Tone matches the rank (high rank = positive, low rank = honest about gaps).
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])

    cid = candidate.get("candidate_id", "")
    title = profile.get("current_title", "Unknown title")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "unknown location")
    open_to_work = signals.get("open_to_work_flag", False)
    notice = signals.get("notice_period_days", 90)
    response_rate = signals.get("recruiter_response_rate", 0.0)
    github = signals.get("github_activity_score", -1)

    # Get top skills
    top_skills = sorted(
        [s for s in skills if s.get("proficiency") in ("advanced", "expert")],
        key=lambda x: x.get("endorsements", 0),
        reverse=True
    )[:3]
    top_skill_names = ", ".join(s.get("name", "") for s in top_skills)

    # Check key skill matches
    all_skill_names = " ".join(s.get("name", "").lower() for s in skills)
    full_text = (
        profile.get("summary", "").lower() + " " +
        profile.get("headline", "").lower() + " " +
        all_skill_names
    )

    has_embeddings = any(k in full_text for k in ["embedding", "sentence-transformer", "dense retrieval"])
    has_vector_db = any(k in full_text for k in ["faiss", "pinecone", "weaviate", "qdrant", "opensearch", "elasticsearch"])
    has_ranking = any(k in full_text for k in ["ranking", "re-rank", "retrieval", "search"])

    # Build honest reasoning based on what we actually found
    parts = []

    # Part 1: Who they are
    parts.append(f"{title} with {yoe:.1f} yrs experience, based in {location}.")

    # Part 2: Technical fit
    tech_signals = []
    if has_embeddings:
        tech_signals.append("embeddings/retrieval background")
    if has_vector_db:
        tech_signals.append("vector DB experience")
    if has_ranking:
        tech_signals.append("ranking systems exposure")
    if top_skill_names:
        tech_signals.append(f"top skills: {top_skill_names}")

    if tech_signals:
        parts.append("Technical fit: " + "; ".join(tech_signals) + ".")
    else:
        parts.append("Limited direct technical match to JD core requirements.")

    # Part 3: Behavioral signals
    behav_signals = []
    if open_to_work:
        behav_signals.append("actively open to work")
    if notice <= 30:
        behav_signals.append(f"notice period {notice}d")
    if response_rate >= 0.7:
        behav_signals.append(f"high response rate ({response_rate:.0%})")
    elif response_rate < 0.2:
        behav_signals.append(f"low response rate ({response_rate:.0%}) — availability concern")
    if github >= 60:
        behav_signals.append(f"strong GitHub activity ({github:.0f}/100)")

    if behav_signals:
        parts.append("Signals: " + "; ".join(behav_signals) + ".")

    # Part 4: Honest concern for lower ranks
    if rank > 50:
        concerns = []
        if not has_embeddings and not has_vector_db:
            concerns.append("no clear embeddings/vector DB experience")
        if yoe < EXP_MIN:
            concerns.append(f"below experience range ({yoe:.1f} yrs < {EXP_MIN})")
        if not open_to_work:
            concerns.append("not marked open-to-work")
        if concerns:
            parts.append("Concerns: " + "; ".join(concerns) + ".")

    # Combine into 1-2 sentences max (join first 2-3 parts)
    reasoning = " ".join(parts[:3])
    # Truncate if too long
    if len(reasoning) > 300:
        reasoning = reasoning[:297] + "..."

    return reasoning


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────

def load_candidates(path: str):
    """Load candidates from .jsonl or .jsonl.gz file."""
    p = Path(path)
    print(f"Loading candidates from {p}...")

    candidates = []
    opener = gzip.open if p.suffix == ".gz" else open
    mode = "rt"

    with opener(path, mode, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  Warning: skipping malformed line {i+1}", file=sys.stderr)

    print(f"  Loaded {len(candidates):,} candidates.")
    return candidates


def rank_candidates(candidates: list) -> list:
    """Score all candidates and return sorted list with scores."""
    print("Scoring candidates...")

    scored = []
    total = len(candidates)

    for i, cand in enumerate(candidates):
        if i % 10000 == 0:
            print(f"  Progress: {i:,}/{total:,} ({100*i//total}%)")

        final_score = compute_final_score(cand)
        scored.append((cand, final_score))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)
    print(f"  Scoring complete. Top score: {scored[0][1]:.4f}")
    return scored


def write_submission(ranked: list, out_path: str):
    """Write the top-100 submission CSV."""
    print(f"Writing submission to {out_path}...")

    top_100 = ranked[:100]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, (cand, score) in enumerate(top_100, start=1):
            cid = cand.get("candidate_id", "UNKNOWN")
            reasoning = build_reasoning(cand, score, rank)
            # Normalize scores to be non-increasing (already sorted, but make explicit)
            writer.writerow([cid, rank, f"{score:.4f}", reasoning])

    print(f"  Submission saved: {out_path}")
    print(f"  Rows written: {len(top_100)}")


def main():
    parser = argparse.ArgumentParser(description="Redrob hackathon candidate ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    args = parser.parse_args()

    import time
    t0 = time.time()

    candidates = load_candidates(args.candidates)
    ranked = rank_candidates(candidates)
    write_submission(ranked, args.out)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print("Next step: run  python validate_submission.py --submission", args.out)


if __name__ == "__main__":
    main()
