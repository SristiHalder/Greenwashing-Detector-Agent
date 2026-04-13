# agent/tavily_client.py

import time
import requests
from agent.config import TAVILY_API_KEY

# Retry configuration
MAX_RETRIES = 3
BACKOFF_BASE = 2  # seconds


def tavily_search(query, max_results=5, search_depth="advanced"):
    """
    Search Tavily with retry logic, timeout, and empty-result logging.

    Args:
        query:        search string
        max_results:  number of results to request
        search_depth: "basic" or "advanced" — advanced returns richer content,
                      which matters for disclosure and external criticism queries

    Returns:
        list of result dicts (may be empty on persistent failure)
    """

    url = "https://api.tavily.com/search"

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,
        "include_raw_content": False  # raw content balloons token usage; snippets are enough
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=(5, 30)  # (connect timeout, read timeout)
            )
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])

            if not results:
                print(f"  ⚠️  Tavily returned 0 results for: '{query}'")

            return results

        except requests.exceptions.Timeout:
            print(f"  ⚠️  Tavily timeout (attempt {attempt}/{MAX_RETRIES}): '{query}'")

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "?"
            print(f"  ⚠️  Tavily HTTP {status} (attempt {attempt}/{MAX_RETRIES}): '{query}'")

            # Don't retry on 4xx except 429 (rate limit)
            if status != 429 and str(status).startswith("4"):
                break

        except requests.exceptions.RequestException as e:
            print(f"  ⚠️  Tavily request error (attempt {attempt}/{MAX_RETRIES}): {e}")

        if attempt < MAX_RETRIES:
            sleep_time = BACKOFF_BASE ** attempt
            print(f"     Retrying in {sleep_time}s...")
            time.sleep(sleep_time)

    print(f"  ✗ Tavily failed after {MAX_RETRIES} attempts for: '{query}'")
    return []