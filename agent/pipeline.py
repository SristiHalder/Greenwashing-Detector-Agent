# agent/pipeline.py
#
# Self-contained pipeline function for use by both main.py (CLI) and app.py (Streamlit).
# No input() calls — validation result is returned to the caller to handle.

import os
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from agent.tavily_client import tavily_search
from agent.extract import fetch_text_from_url
from agent.text_splitter import safe_make_chunks
from agent.ai_chunks import run_ai_chunking
from agent.company_validator import company_validator
from agent.scoring import compute_greenwashing_score
from agent.report_generator import generate_report

CHUNK_MAX_CHARS = 6000
CHUNK_OVERLAP_CHARS = 500

CATEGORY_TARGETS = {"marketing": 10, "disclosure": 10, "external": 10}

SEARCH_QUERIES = {
    "marketing": [
        "{company} sustainability",
        "{company} net zero goals",
        "{company} climate commitments",
        "{company} sustainability strategy",
    ],
    "disclosure": [
        "{company} sustainability report",
        "{company} ESG report",
        "{company} annual sustainability data",
        "{company} impact report",
    ],
    "external": [
        "{company} greenwashing",
        "{company} environmental controversy",
        "{company} climate lawsuit",
        "{company} sustainability criticism",
    ],
}


def run_full_pipeline(company, on_status=None):
    """
    Run the complete greenwashing detection pipeline.

    Args:
        company:    company name string
        on_status:  optional callback(str) called with progress messages

    Returns a dict with one of these shapes:
        {"status": "ambiguous", "reason": str}
        {"status": "invalid",   "reason": str}
        {"status": "error",     "reason": str}
        {"status": "success",   "company": str, "score": dict,
                                "report": str,  "ai_chunks": list}
    """

    def log(msg):
        if on_status:
            on_status(msg)
        print(msg)

    company = company.strip()
    company_dir = os.path.join("data", company.lower())
    os.makedirs(company_dir, exist_ok=True)

    # ── 1. Validate ──────────────────────────────────────────────────────────
    log("🔍 Validating company identity...")
    validation = company_validator(company)

    with open(os.path.join(company_dir, "validation.json"), "w") as f:
        json.dump(validation, f, indent=2)

    if validation["status"] == "ambiguous":
        return {"status": "ambiguous", "reason": validation["reason"]}
    if validation["status"] == "invalid":
        return {"status": "invalid", "reason": validation["reason"]}

    log(f"  ✓ {validation['reason']}")

    # ── 2. Search ─────────────────────────────────────────────────────────────
    log("🔎 Collecting sources...")
    category_urls = {"marketing": [], "disclosure": [], "external": []}
    seen = set()

    for category, query_templates in SEARCH_QUERIES.items():
        for template in query_templates:
            if len(category_urls[category]) >= CATEGORY_TARGETS[category]:
                break
            q = template.format(company=company)
            log(f"  Query: {q}")
            results = tavily_search(q, max_results=5)
            for r in results:
                url = r["url"]
                if url in seen:
                    continue
                seen.add(url)
                category_urls[category].append({"category": category, "url": url})
                if len(category_urls[category]) >= CATEGORY_TARGETS[category]:
                    break
        log(f"  ✓ {category}: {len(category_urls[category])} URLs")

    unique_urls = []
    for cat in category_urls:
        unique_urls.extend(category_urls[cat])

    if not unique_urls:
        return {"status": "error", "reason": "No URLs collected from search."}

    log(f"✓ Total URLs collected: {len(unique_urls)}")

    # ── 3. Extract ────────────────────────────────────────────────────────────
    EXTRACT_TIMEOUT = 15  # seconds per source

    log("📄 Extracting text from sources...")
    processed_sources = []

    for i, u in enumerate(unique_urls, 1):
        log(f"  ({i}/{len(unique_urls)}) {u['url']}")
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(fetch_text_from_url, u["url"])
                text, fmt = future.result(timeout=EXTRACT_TIMEOUT)
        except FuturesTimeoutError:
            log(f"    ✗ Timed out after {EXTRACT_TIMEOUT}s — skipped")
            continue
        except Exception as e:
            log(f"    ✗ Failed ({e})")
            continue
        if not text or len(text) < 800:
            log("    ✗ Too short — skipped")
            continue
        log(f"    ✓ {len(text)} chars [{fmt}]")
        processed_sources.append({
            "company": company,
            "category": u["category"],
            "url": u["url"],
            "source_format": fmt,
            "text": text,
        })

    if not processed_sources:
        return {"status": "error", "reason": "No valid sources could be extracted."}

    log(f"✓ Valid sources: {len(processed_sources)}")

    # ── 4. Chunk ──────────────────────────────────────────────────────────────
    log("✂️  Chunking text...")
    chunks_out = []
    chunk_id = 0

    for s in processed_sources:
        chunks = safe_make_chunks(
            s["text"],
            max_chars=CHUNK_MAX_CHARS,
            overlap_chars=CHUNK_OVERLAP_CHARS,
            timeout=40,
        )
        for i, ch in enumerate(chunks):
            chunk_id += 1
            chunks_out.append({
                "chunk_id": f"{company}_{chunk_id:05d}",
                "company": company,
                "category": s["category"],
                "source_format": s["source_format"],
                "source_url": s["url"],
                "chunk_index_in_source": i,
                "chunk_text": ch,
            })

    if not chunks_out:
        return {"status": "error", "reason": "No chunks created from extracted text."}

    with open(os.path.join(company_dir, "chunks.jsonl"), "w") as f:
        for c in chunks_out:
            f.write(json.dumps(c) + "\n")

    log(f"✓ Total chunks: {len(chunks_out)}")

    # ── 5. AI Analysis ────────────────────────────────────────────────────────
    log("🤖 Running AI semantic analysis (parallel)...")
    run_ai_chunking(company)

    # ── 6. Score ──────────────────────────────────────────────────────────────
    log("📊 Computing greenwashing score...")
    compute_greenwashing_score(company)

    score_path = os.path.join(company_dir, "score.json")
    if not os.path.exists(score_path):
        return {"status": "error", "reason": "Scoring step did not produce output."}

    with open(score_path, "r") as f:
        score_data = json.load(f)

    if score_data.get("integrity_score") is None:
        return {"status": "error", "reason": "Insufficient data — too few relevant chunks were found."}

    log(f"  ✓ Integrity Score: {score_data['integrity_score']} / 100 — {score_data['risk_label']}")

    # ── 7. Report ─────────────────────────────────────────────────────────────
    log("📝 Generating forensic report...")
    report_text = generate_report(company)

    if not report_text:
        return {"status": "error", "reason": "Report generation failed."}

    # ── 8. Load AI chunks for Q&A ─────────────────────────────────────────────
    ai_chunks_path = os.path.join(company_dir, "ai_chunks.jsonl")
    ai_chunks = []
    if os.path.exists(ai_chunks_path):
        with open(ai_chunks_path, "r", encoding="utf-8") as f:
            ai_chunks = [json.loads(line) for line in f]

    log("✅ Pipeline complete!")

    return {
        "status": "success",
        "company": company,
        "score": score_data,
        "report": report_text,
        "ai_chunks": ai_chunks,
    }