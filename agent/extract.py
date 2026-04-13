# agent/extract.py

import io
import requests
from bs4 import BeautifulSoup
import pdfplumber

# Hard cap: skip PDFs larger than 50 MB to avoid memory issues
PDF_MAX_BYTES = 50 * 1024 * 1024

# HTML tags that produce noise on corporate sustainability pages
NOISE_TAGS = [
    "script", "style", "nav", "footer", "header",
    "aside", "form", "button", "noscript", "iframe",
    "figure"  # often caption-only, low signal
]

# CSS class/id fragments that strongly indicate cookie banners, popups, ads
NOISE_CLASS_FRAGMENTS = ["cookie", "banner", "popup", "modal", "overlay", "gdpr", "consent"]


def _strip_noise(soup):
    """Remove tags and elements that add noise to extracted text."""

    # Remove by tag name
    for tag in soup(NOISE_TAGS):
        tag.decompose()

    # Remove by class/id fragment (cookie banners, popups, etc.)
    for fragment in NOISE_CLASS_FRAGMENTS:
        for el in soup.find_all(
            attrs={"class": lambda c: c and any(fragment in cls.lower() for cls in c)}
        ):
            el.decompose()
        for el in soup.find_all(
            attrs={"id": lambda i: i and fragment in i.lower()}
        ):
            el.decompose()


def fetch_html_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        _strip_noise(soup)

        text = soup.get_text(separator="\n", strip=True)
        return text

    except Exception as e:
        print(f"  HTML error: {e}")
        return ""


def fetch_pdf_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}

        # Stream the download so we can check size before loading into memory
        with requests.get(url, headers=headers, timeout=30, stream=True) as response:
            response.raise_for_status()

            # Check Content-Length header if present
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > PDF_MAX_BYTES:
                print(f"  ✗ PDF too large ({int(content_length) // (1024*1024)} MB) — skipping: {url}")
                return ""

            # Read in chunks, enforcing size cap
            chunks = []
            total = 0
            for chunk in response.iter_content(chunk_size=1024 * 256):
                total += len(chunk)
                if total > PDF_MAX_BYTES:
                    print(f"  ✗ PDF exceeded {PDF_MAX_BYTES // (1024*1024)} MB cap — skipping: {url}")
                    return ""
                chunks.append(chunk)

            raw = b"".join(chunks)

        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)

        return "\n".join(pages)

    except Exception as e:
        print(f"  PDF error: {e}")
        return ""


def fetch_text_from_url(url):
    """
    Fetch and extract text from a URL.

    Detection order:
      1. URL ends with .pdf  → PDF parser
      2. Fetch as HTML, check Content-Type header → PDF parser if application/pdf
      3. Otherwise treat as HTML

    Returns:
        (text: str, format: str)  where format is "pdf", "html", or "unknown"
    """

    url_lower = url.lower()

    # Fast path: URL extension is .pdf
    if url_lower.endswith(".pdf"):
        text = fetch_pdf_text(url)
        return text, "pdf"

    # Fetch HTML — but check Content-Type in case the server serves a PDF
    # without a .pdf extension (common for ESG report download links)
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        head_response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
        content_type = head_response.headers.get("Content-Type", "")

        if "application/pdf" in content_type:
            text = fetch_pdf_text(url)
            return text, "pdf"

    except Exception:
        pass  # HEAD request failed — fall through to HTML fetch

    text = fetch_html_text(url)

    if text and len(text) > 500:
        return text, "html"

    return "", "unknown"