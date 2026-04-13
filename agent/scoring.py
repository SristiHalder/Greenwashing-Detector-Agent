# agent/scoring.py

import os
import json
from collections import Counter


# -------------------------------------------------
# Utility
# -------------------------------------------------

def clamp(value, min_value=0, max_value=100):
    return max(min_value, min(max_value, value))


# -------------------------------------------------
# Main Scoring Function
# -------------------------------------------------

def compute_greenwashing_score(company):

    company_dir = os.path.join("data", company.lower())
    ai_chunks_path = os.path.join(company_dir, "ai_chunks.jsonl")

    if not os.path.exists(ai_chunks_path):
        print("No AI chunks found. Cannot compute score.")
        return

    with open(ai_chunks_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    if not records:
        print("No relevant AI records found. Writing empty score.")
        empty_result = {
            "company": company,
            "integrity_score": None,
            "confidence_score": 0,
            "risk_label": "Insufficient Data",
            "metrics": {
                "total_chunks": 0,
                "error": "No relevant chunks were extracted. Check AI chunking output."
            }
        }
        score_path = os.path.join(company_dir, "score.json")
        with open(score_path, "w", encoding="utf-8") as f:
            json.dump(empty_result, f, indent=2)
        print("⚠️  score.json written with 'Insufficient Data' status.\n")
        return

    total_chunks = len(records)

    # -------------------------------------------------
    # Aggregate Metrics
    # -------------------------------------------------

    verdict_counter = Counter()
    red_flag_counter = Counter()
    specificity_counter = Counter()
    confidence_scores = []
    categories = set()

    for r in records:
        ai = r["ai_analysis"]

        verdict_counter[ai.get("verdict")] += 1
        specificity_counter[ai.get("specificity")] += 1

        for flag in ai.get("red_flags", []):
            red_flag_counter[flag] += 1

        confidence_scores.append(ai.get("confidence", 0))
        categories.add(r.get("category"))

    claims = verdict_counter["claim"]
    disclosures = verdict_counter["disclosure"]
    external = verdict_counter["external_critique"]

    vague = specificity_counter["vague"]
    measurable = specificity_counter["measurable"]

    # -------------------------------------------------
    # Ratios (0–1)
    # -------------------------------------------------

    # Specificity ratios are computed over claims only.
    # Guard: if claims are too few, clarity_score is unreliable — zero it out.
    MIN_CLAIMS_FOR_CLARITY = 5

    if claims >= MIN_CLAIMS_FOR_CLARITY:
        vague_ratio = vague / claims
        measurable_ratio = measurable / claims
        clarity_reliable = True
    else:
        vague_ratio = 0
        measurable_ratio = 0
        clarity_reliable = False

    transparency_ratio = disclosures / total_chunks

    # Fix: controversy_ratio uses only external-category chunk count to avoid
    # double-counting disclosure chunks that happen to have a controversy flag.
    external_category_chunks = sum(
        1 for r in records if r.get("category") == "external"
    )
    controversy_ratio = (
        red_flag_counter["controversy"]
        + red_flag_counter["legal_action"]
        + external_category_chunks
    ) / total_chunks

    offset_ratio = red_flag_counter["offset_heavy"] / total_chunks
    scope3_ratio = red_flag_counter["missing_scope3"] / total_chunks

    # Fix: marketing_disclosure_gap now compares like-for-like.
    # marketing_chunks = chunks from the marketing category
    # disclosure_chunks = chunks from the disclosure category
    # Both are raw chunk counts from the same pool.
    marketing_chunks = sum(
        1 for r in records if r.get("category") == "marketing"
    )
    disclosure_chunks = sum(
        1 for r in records if r.get("category") == "disclosure"
    )

    marketing_disclosure_gap = abs(marketing_chunks - disclosure_chunks) / total_chunks

    avg_ai_confidence = (
        sum(confidence_scores) / len(confidence_scores)
        if confidence_scores else 0
    )

    # -------------------------------------------------
    # Subscores (0–1)
    # -------------------------------------------------

    if clarity_reliable:
        clarity_score = clamp(measurable_ratio - vague_ratio, -1, 1)
        clarity_score = (clarity_score + 1) / 2  # normalize to 0–1
    else:
        # Not enough claims to assess — treat as neutral (0.5)
        clarity_score = 0.5

    transparency_score = clamp(transparency_ratio, 0, 1)
    balance_score = clamp(1 - marketing_disclosure_gap, 0, 1)

    controversy_penalty = clamp(controversy_ratio, 0, 1)
    offset_penalty = clamp(offset_ratio, 0, 1)
    scope3_penalty = clamp(scope3_ratio, 0, 1)

    # -------------------------------------------------
    # Integrity Score
    # Weights: +0.30 +0.20 +0.20 -0.15 -0.10 -0.05 = net 0.40 positive max.
    # Score is computed in 0–1 space then scaled to 0–100.
    # A perfect company (all positives max, all penalties zero) scores 0.70 → 70/100 raw,
    # which after clamp maps to 70. This is intentional: the scale rewards
    # strong positive signals but does not allow a 100 without zero controversy.
    # -------------------------------------------------

    integrity_raw = (
        0.30 * clarity_score
        + 0.20 * transparency_score
        + 0.20 * balance_score
        - 0.15 * controversy_penalty
        - 0.10 * offset_penalty
        - 0.05 * scope3_penalty
    )

    # Rescale from [0, 0.70] to [0, 100] so the full range is usable
    integrity_score = clamp(round((integrity_raw / 0.70) * 100, 2), 0, 100)

    # -------------------------------------------------
    # Confidence Score
    # -------------------------------------------------

    volume_score = min(1, total_chunks / 150)
    diversity_score = len(categories) / 3  # marketing, disclosure, external
    ai_conf_score = avg_ai_confidence

    confidence_score = (
        0.5 * volume_score
        + 0.3 * diversity_score
        + 0.2 * ai_conf_score
    )

    confidence_score = clamp(round(confidence_score * 100, 2), 0, 100)

    # -------------------------------------------------
    # Risk Label
    # -------------------------------------------------

    if integrity_score >= 80:
        label = "High Integrity"
    elif integrity_score >= 60:
        label = "Moderate Integrity"
    elif integrity_score >= 40:
        label = "Weak Integrity"
    else:
        label = "Likely Greenwashing"

    # -------------------------------------------------
    # Output Object
    # -------------------------------------------------

    result = {
        "company": company,
        "integrity_score": integrity_score,
        "confidence_score": confidence_score,
        "risk_label": label,
        "metrics": {
            "total_chunks": total_chunks,
            "claims": claims,
            "disclosures": disclosures,
            "external_critique": external,
            "clarity_reliable": clarity_reliable,
            "vague_ratio": round(vague_ratio, 3),
            "measurable_ratio": round(measurable_ratio, 3),
            "transparency_ratio": round(transparency_ratio, 3),
            "marketing_chunks": marketing_chunks,
            "disclosure_chunks": disclosure_chunks,
            "marketing_disclosure_gap": round(marketing_disclosure_gap, 3),
            "controversy_ratio": round(controversy_ratio, 3),
            "offset_ratio": round(offset_ratio, 3),
            "scope3_ratio": round(scope3_ratio, 3),
            "avg_ai_confidence": round(avg_ai_confidence, 3)
        }
    }

    # Save score
    score_path = os.path.join(company_dir, "score.json")
    with open(score_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # -------------------------------------------------
    # Print Clean Output
    # -------------------------------------------------

    print("\n===== GREENWASHING INTEGRITY REPORT =====\n")
    print(f"Company:         {company}")
    print(f"Integrity Score: {integrity_score} / 100")
    print(f"Confidence Score:{confidence_score} / 100")
    print(f"Assessment:      {label}")

    if not clarity_reliable:
        print(f"\n  ⚠️  Clarity score set to neutral (fewer than {MIN_CLAIMS_FOR_CLARITY} claim chunks found).")

    print("\nKey Indicators:")
    print(f"  • Measurable Claim Ratio:     {round(measurable_ratio, 2)}")
    print(f"  • Vague Claim Ratio:          {round(vague_ratio, 2)}")
    print(f"  • Transparency Ratio:         {round(transparency_ratio, 2)}")
    print(f"  • Marketing–Disclosure Gap:   {round(marketing_disclosure_gap, 2)}")
    print(f"  • Controversy Ratio:          {round(controversy_ratio, 2)}")
    print("\nFull breakdown saved to score.json\n")