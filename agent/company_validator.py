# agent/company_validator.py

import re
from collections import Counter
from urllib.parse import urlparse
from agent.tavily_client import tavily_search


TRUSTED_DOMAINS = [
    "wikipedia.org", "bloomberg.com", "reuters.com", "sec.gov",
    "linkedin.com", "forbes.com", "ft.com", "nytimes.com",
    "yahoo.com", "finance.yahoo.com", "britannica.com"
]

COMPANY_INDICATORS = ["inc", "ltd", "corp", "company", "plc", "holdings", "group", "ag", "sa", "gmbh"]

MIN_CONFIDENCE_VALID = 0.25
MIN_RESULTS_REQUIRED = 3


def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_root_domain(url):
    try:
        parsed = urlparse(url)
        return parsed.netloc.replace("www.", "")
    except Exception:
        return ""


def company_validator(company):

    queries = [
        f"{company} company",
        f"{company} official website",
        f"{company} headquarters"
    ]

    results = []
    for q in queries:
        res = tavily_search(q, max_results=5)
        results.extend(res)

    company_lower = normalize(company)

    # FIX: fallback if input is too short or weird
    if not company_lower or len(company_lower) < 2:
        return {
            "status": "invalid",
            "confidence": 0.0,
            "reason": "Please enter a valid company name.",
            "official_domain": None
        }

    company_words = [w for w in company_lower.split() if len(w) > 2]

    # TOO FEW RESULTS
    if len(results) < MIN_RESULTS_REQUIRED:
        return {
            "status": "invalid",
            "confidence": 0.0,
            "reason": "Not enough information found. Try a more specific company name.",
            "official_domain": None
        }

    match_count = 0
    trusted_count = 0
    domains = []

    for r in results:
        title = normalize(r.get("title", ""))
        content = normalize(r.get("content", ""))
        url = r.get("url", "")

        root_domain = extract_root_domain(url)
        if root_domain:
            domains.append(root_domain)

        # IMPROVED MATCHING
        if (
            any(word in title or word in content for word in company_words)
            or company_lower in title
        ):
            match_count += 1

        if root_domain in TRUSTED_DOMAINS:
            trusted_count += 1

    match_ratio = match_count / len(results)
    trusted_ratio = trusted_count / len(results)

    domain_counts = Counter(domains)

    if domain_counts:
        top_domain, top_count = domain_counts.most_common(1)[0]
        top_domain_share = top_count / len(domains)
    else:
        top_domain = None
        top_domain_share = 0

    unique_domain_count = len(domain_counts)

    confidence = round(
        (0.6 * match_ratio) + (0.4 * trusted_ratio),
        2
    )

    # AMBIGUOUS
    if (
        len(company_words) <= 2
        and unique_domain_count >= 6
        and top_domain_share < 0.25
        and match_ratio < 0.6 
    ):
        return {
            "status": "ambiguous",
            "confidence": confidence,
            "reason": "This name is too broad or ambiguous. Try a more specific company name."
        }

    # INVALID
    if confidence < MIN_CONFIDENCE_VALID:
        return {
            "status": "invalid",
            "confidence": confidence,
            "reason": "Could not confidently identify a company. Try a clearer or full company name."
        }

    #  VALID
    return {
        "status": "valid",
        "confidence": confidence,
        "reason": "Company appears valid.",
        "official_domain": top_domain
    }
