import os
import json
from openai import OpenAI
from agent.config import OPENAI_API_KEY
from agent.utils import select_key_chunks

client = OpenAI(api_key=OPENAI_API_KEY)

REPORT_FILENAME = "report.md"


# --------------------------------------------------
# Load Data
# --------------------------------------------------

def load_company_data(company):
    company_dir = os.path.join("data", company.lower())

    with open(os.path.join(company_dir, "score.json"), "r") as f:
        score_data = json.load(f)

    with open(os.path.join(company_dir, "ai_chunks.jsonl"), "r") as f:
        chunks = [json.loads(line) for line in f]

    return score_data, chunks


# --------------------------------------------------
# Build Prompt (FIXED)
# --------------------------------------------------

def build_prompt(score_data, key_chunks):

    integrity_score = score_data["integrity_score"]

    # ✅ FIX: DO NOT multiply by 100 again
    confidence = round(score_data["confidence_score"], 2)

    label = score_data["risk_label"]

    return f"""
You are an ESG forensic analyst.

Below is structured scoring data for a company.

Integrity Score: {integrity_score}
Confidence Score: {confidence}
Risk Label: {label}

Metrics:
{json.dumps(score_data["metrics"], indent=2)}

Key Evidence:
{json.dumps(key_chunks, indent=2)}

Produce a detailed investigative narrative report structured EXACTLY as follows:

---
INTEGRITY SCORE: {integrity_score} / 100
ASSESSMENT: {label}
CONFIDENCE: {confidence} / 100
---

Then write sections:
1. Executive Summary
2. Key Metrics Analysis
3. Detailed Explanation of Evidence
4. Connection Between Evidence and Score
5. Governance & Disclosure Risks
6. What Would Reduce Risk Score
7. Source Citations (include URLs inline)

Rules:
- The score block above MUST appear at the very top of your response.
- DO NOT change or recompute the score.
- DO NOT introduce any new numbers for score.
- DO NOT rescale confidence.
- Use ONLY the provided score_data.
- Write in clear professional prose.
- Be specific and analytical.
"""


# --------------------------------------------------
# Generate Report
# --------------------------------------------------

def generate_report(company):

    score_data, chunks = load_company_data(company)

    if score_data.get("integrity_score") is None:
        print("⚠️ No score data available. Skipping report.")
        return None

    key_chunks = select_key_chunks(chunks, limit=8)

    prompt = build_prompt(score_data, key_chunks)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You produce forensic ESG risk reports."},
            {"role": "user", "content": prompt}
        ]
    )

    report_text = response.choices[0].message.content

    company_dir = os.path.join("data", company.lower())
    report_path = os.path.join(company_dir, REPORT_FILENAME)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n================ ESG FORENSIC REPORT ================\n")
    print(report_text)
    print("\n=====================================================\n")

    return report_text