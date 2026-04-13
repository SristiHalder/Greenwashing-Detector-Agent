import os
import json

from agent.tavily_client import tavily_search
from agent.extract import fetch_text_from_url
from agent.text_splitter import safe_make_chunks
from agent.ai_chunks import run_ai_chunking
from agent.company_validator import company_validator
from agent.scoring import compute_greenwashing_score
from agent.report_generator import generate_report
from agent.explain_agent import answer_question

# DB
from agent.db import get_company, save_company


CHUNK_MAX_CHARS = 6000
CHUNK_OVERLAP_CHARS = 500


company = input("Enter company name: ").strip()

if not company:
    print("No company name provided.")
    exit()


# =========================================================
# CHECK SUPABASE CACHE
# =========================================================

print("\n--- Checking Supabase Cache ---\n")

existing = get_company(company)

if existing:
    print("⚡ Found existing analysis in Supabase!")
    print("Skipping pipeline.\n")

    print("Integrity Score:", existing.get("integrity_score"))
    print("Confidence Score:", existing.get("confidence_score"))
    print("Risk Label:", existing.get("risk_label"))

    print("\n--- REPORT ---\n")
    print(existing.get("report_text"))

    exit()


# =========================================================
# RUN FULL PIPELINE
# =========================================================

print("\nNo cached data found. Running full pipeline...\n")


# ---------------------------
# Validation
# ---------------------------

while True:
    print("\n--- Validating Company ---\n")

    validation = company_validator(company)

    print("Status:", validation["status"])
    print("Confidence:", validation["confidence"])
    print("Reason:", validation["reason"])

    if validation["status"] == "valid":
        print("✓ Company validated.\n")
        break

    if validation["status"] == "invalid":
        company = input("\nInvalid. Enter different name (or Enter to exit): ").strip()
        if not company:
            exit()
        continue

    if validation["status"] == "ambiguous":
        company = input("\nAmbiguous. Enter FULL official company name: ").strip()
        if not company:
            exit()
        continue


# ---------------------------
# Search
# ---------------------------

CATEGORY_TARGETS = {
    "marketing": 10,
    "disclosure": 10,
    "external": 10
}

search_queries = {
    "marketing": [
        f"{company} sustainability",
        f"{company} net zero goals",
        f"{company} climate commitments",
        f"{company} sustainability strategy"
    ],
    "disclosure": [
        f"{company} sustainability report",
        f"{company} ESG report",
        f"{company} annual sustainability data",
        f"{company} impact report"
    ],
    "external": [
        f"{company} greenwashing",
        f"{company} environmental controversy",
        f"{company} climate lawsuit",
        f"{company} sustainability criticism"
    ]
}

print("\n--- Collecting Sources ---\n")

category_urls = {"marketing": [], "disclosure": [], "external": []}
seen = set()

for category, queries in search_queries.items():
    print(f"\nSearching category: {category}")

    for q in queries:
        if len(category_urls[category]) >= CATEGORY_TARGETS[category]:
            break

        results = tavily_search(q, max_results=5)

        for r in results:
            url = r["url"]

            if url in seen:
                continue

            seen.add(url)

            category_urls[category].append({
                "category": category,
                "url": url
            })

            if len(category_urls[category]) >= CATEGORY_TARGETS[category]:
                break

unique_urls = []
for cat in category_urls:
    unique_urls.extend(category_urls[cat])

if not unique_urls:
    print("No URLs collected.")
    exit()


# ---------------------------
# Extraction
# ---------------------------

processed_sources = []

for u in unique_urls:
    try:
        text, fmt = fetch_text_from_url(u["url"])
    except Exception:
        continue

    if not text or len(text) < 800:
        continue

    processed_sources.append({
        "company": company,
        "category": u["category"],
        "url": u["url"],
        "source_format": fmt,
        "text": text
    })

if not processed_sources:
    print("No valid sources.")
    exit()


# ---------------------------
# Chunking
# ---------------------------

chunks_out = []

for s in processed_sources:
    chunks = safe_make_chunks(
        s["text"],
        max_chars=CHUNK_MAX_CHARS,
        overlap_chars=CHUNK_OVERLAP_CHARS
    )

    for i, ch in enumerate(chunks):
        chunks_out.append({
            "company": company,
            "category": s["category"],
            "source_url": s["url"],
            "chunk_text": ch
        })

if not chunks_out:
    print("No chunks created.")
    exit()


# ---------------------------
# AI Chunking
# ---------------------------

run_ai_chunking(company)


# =========================================================
# SCORE + REPORT
# =========================================================

compute_greenwashing_score(company)

report = generate_report(company)


# =========================================================
# SAVE TO SUPABASE (FIXED)
# =========================================================

print("\n--- Saving Results to Supabase ---\n")

# ✅ LOAD REAL SCORE (THIS FIXES EVERYTHING)
with open(os.path.join("data", company.lower(), "score.json"), "r") as f:
    score_data = json.load(f)

ai_chunks = []

success = save_company(company, score_data, report, ai_chunks)

if success:
    print("Saved to Supabase\n")
else:
    print("Failed to save.\n")

print("Agent pipeline complete.\n")