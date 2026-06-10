# Redrob Hackathon — Intelligent Candidate Ranking System

**India Runs Hackathon · Track 1: Data & AI Challenge**

## What this does

This system ranks 100,000 candidate profiles against the **Senior AI Engineer — Founding Team** job description, outputting a top-100 CSV with scores and reasoning.

It is designed to run **entirely on CPU, with no external API calls, in under 5 minutes** for 100,000 candidates — meeting the competition's compute constraints exactly.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

No heavy ML libraries required — the core ranker uses only Python's standard library.

### 2. Run the ranker

```bash
python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv
```

Or with the uncompressed file:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

### 3. Validate before submitting

```bash
python validate_submission.py --submission ./submission.csv
```

---

## How it works

The system scores each candidate across 5 dimensions and combines them into a single score:

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| Skill match | 35% | Match against required/nice-to-have skills from JD |
| Experience fit | 20% | Ideal 6-8 year range; penalties outside 5-9 |
| Title relevance | 15% | ML/AI/NLP engineer titles vs non-technical titles |
| Behavioral signals | 20% | Redrob engagement, availability, response rate |
| Location match | 10% | Pune/Noida/Delhi NCR/Hyderabad/Mumbai preference |

### Key design decisions

**Why not just keyword match?**
The JD explicitly warns against this. A candidate with "RAG" and "Pinecone" in their skills but a Marketing Manager title should rank below a candidate whose career history shows shipping a recommendation system, even without the exact keywords.

**Honeypot detection**
The dataset contains ~80 impossible profiles (expert in 10 skills with 0 months used, etc.). Our `honeypot_penalty()` function detects and zeros these out.

**Behavioral signals as a multiplier**
A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is not actually hireable. The behavioral score penalises such candidates and rewards those who are actively engaged.

**No GPU, no API calls**
All scoring is pure Python arithmetic over structured JSON fields. 100,000 candidates score in ~60-90 seconds on a standard laptop CPU.

---

## File structure

```
redrob_ranker/
├── rank.py              # Main ranker — produces submission.csv
├── app.py               # Streamlit sandbox demo (hosted on Streamlit Cloud)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Sandbox / Demo

Live demo on Streamlit Cloud:  
👉 **[your-streamlit-link-here]** *(deploy via streamlit.io/cloud after pushing to GitHub)*

Upload any subset of `candidates.jsonl` (up to 100 candidates) and get a ranked CSV output instantly.

---

## Compute environment

- Python 3.10+
- No GPU required
- 16 GB RAM sufficient
- Runtime: ~60-90 seconds for 100,000 candidates

## AI tools declaration

Built with assistance from Claude (Anthropic) for code structure and documentation.
