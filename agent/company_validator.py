# agent/company_validator.py

import re
from collections import Counter
from urllib.parse import urlparse
from agent.tavily_client import tavily_search


TRUSTED_DOMAINS = [
    "wikipedia.org",
    "bloomberg.com",
    "reuters.com",
    "sec.gov",
    "linkedin.com",
    "forbes.com",
    "ft.com",
    "nytimes.com",
    "yahoo.com",
    "finance.yahoo.com",
    "britannica.com"
]

COMPANY_INDICATORS = ["inc", "ltd", "corp", "company", "plc", "holdings", "group", "ag", "sa", "gmbh"]

MIN_CONFIDENCE_VALID = 0.35
MIN_RESULTS_REQUIRED = 5


def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_root_domain(url):
    try:
        parsed = urlparse(url)
        return parsed.netloc.replace("www.", "")
    except Exception:
        return ""


def is_company_wikipedia_page(url, title, company_lower):
    url = url.lower()
    title = title.lower()

    if "wikipedia.org/wiki/" not in url:
        return False

    # URL must contain company name (spaces → underscores in Wikipedia URLs)
    company_slug = company_lower.replace(" ", "_")
    if company_slug not in url:
        return False

    # Title must indicate a company-like entity
    if any(ind in title for ind in COMPANY_INDICATORS):
        return True

    if company_lower in title:
        return True

    return False


def find_official_domain(company_lower, domain_counts):
    """
    Try to find a domain that plausibly belongs to the company.

    Strategy: check ALL words in the company name (not just the first word),
    preferring longer matches to avoid false positives like 'general' matching
    'general-store.com'.
    """

    words = [w for w in company_lower.split() if len(w) > 3]  # skip short words

    # Score each domain by how many company words it contains
    best_domain = None
    best_score = 0

    for domain in domain_counts:
        score = sum(1 for w in words if w in domain)
        if score > best_score:
            best_score = score
            best_domain = domain

    # Fall back to most common domain if no word match found
    if not best_domain and domain_counts:
        best_domain = domain_counts.most_common(1)[0][0]

    return best_domain


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

    if len(results) < MIN_RESULTS_REQUIRED:
        return {
            "status": "invalid",
            "confidence": 0.0,
            "reason": "Insufficient search results.",
            "official_domain": None,
            "evidence": {"total_results": len(results)}
        }

    match_count = 0
    trusted_count = 0
    domains = []
    has_strong_wiki = False

    for r in results:
        title = normalize(r.get("title", ""))
        content = normalize(r.get("content", ""))
        url = r.get("url", "")

        root_domain = extract_root_domain(url)
        if root_domain:
            domains.append(root_domain)

        if company_lower in title or company_lower in content:
            match_count += 1

        if root_domain in TRUSTED_DOMAINS:
            trusted_count += 1

        if is_company_wikipedia_page(url, title, company_lower):
            has_strong_wiki = True

    match_ratio = match_count / len(results)
    trusted_ratio = trusted_count / len(results)
    wiki_bonus = 0.1 if has_strong_wiki else 0.0  # Wikipedia presence is a real signal

    domain_counts = Counter(domains)

    if domain_counts:
        top_domain, top_count = domain_counts.most_common(1)[0]
        top_domain_share = top_count / len(domains)
    else:
        top_domain = None
        top_domain_share = 0

    # Confidence now includes Wikipedia bonus
    confidence = round(
        (0.55 * match_ratio) + (0.35 * trusted_ratio) + wiki_bonus,
        2
    )
    confidence = min(confidence, 1.0)  # cap at 1.0

    unique_domain_count = len(domain_counts)

    # --------------------------------------------------
    # STRONG GLOBAL BRAND OVERRIDE
    # --------------------------------------------------

    if match_ratio >= 0.6 and (trusted_ratio >= 0.3 or has_strong_wiki):
        official_domain = find_official_domain(company_lower, domain_counts)

        return {
            "status": "valid",
            "confidence": confidence,
            "reason": "Strong global brand identity detected.",
            "official_domain": official_domain,
            "evidence": {
                "match_ratio": round(match_ratio, 2),
                "trusted_ratio": round(trusted_ratio, 2),
                "has_wikipedia": has_strong_wiki,
                "top_domain_share": round(top_domain_share, 2)
            }
        }

    # --------------------------------------------------
    # AMBIGUITY DETECTION
    # Trigger for short names (1–2 words) with fragmented domain spread.
    # Multi-word names can be equally ambiguous (e.g. "General Electric").
    # --------------------------------------------------

    if (
        len(company.split()) <= 2
        and unique_domain_count >= 4
        and top_domain_share < 0.35
    ):
        return {
            "status": "ambiguous",
            "confidence": confidence,
            "reason": "Multiple distinct domains detected — name may be ambiguous.",
            "official_domain": None,
            "evidence": {
                "unique_domains": unique_domain_count,
                "top_domain_share": round(top_domain_share, 2),
                "domains": list(domain_counts.keys())[:5]
            }
        }

    # --------------------------------------------------
    # INVALID
    # --------------------------------------------------

    if confidence < MIN_CONFIDENCE_VALID:
        return {
            "status": "invalid",
            "confidence": confidence,
            "reason": "Low identity confidence.",
            "official_domain": None,
            "evidence": {
                "match_ratio": round(match_ratio, 2),
                "trusted_ratio": round(trusted_ratio, 2),
                "has_wikipedia": has_strong_wiki
            }
        }

    # --------------------------------------------------
    # DEFAULT VALID
    # --------------------------------------------------

    official_domain = find_official_domain(company_lower, domain_counts)

    return {
        "status": "valid",
        "confidence": confidence,
        "reason": "Company appears legitimate.",
        "official_domain": official_domain,
        "evidence": {
            "match_ratio": round(match_ratio, 2),
            "trusted_ratio": round(trusted_ratio, 2),
            "has_wikipedia": has_strong_wiki,
            "top_domain_share": round(top_domain_share, 2)
        }
    }