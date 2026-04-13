import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError


# --------------------------------------------------
# Core Chunking Logic (Safe + Bounded)
# --------------------------------------------------

def make_chunks(
    text,
    max_chars=6000,
    overlap_chars=500,
    max_source_chars=200_000,
    max_chunks=200
):
    """
    Production-safe chunking.

    Protections:
    - Caps maximum source size
    - Caps maximum number of chunks
    - Avoids mid-word splits when possible
    """

    if not text:
        return []

    # -----------------------------------------
    # Hard cap source size (prevents huge PDFs)
    # -----------------------------------------

    text = text[:max_source_chars]

    chunks = []
    text_length = len(text)
    start = 0

    while start < text_length:

        if len(chunks) >= max_chunks:
            print("⚠️ Max chunk limit reached. Truncating source.")
            break

        end = start + max_chars

        # Avoid cutting mid-word if possible
        if end < text_length:
            while end > start and text[end] not in [" ", "\n"]:
                end -= 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Move forward with overlap
        start = end - overlap_chars

        if start <= 0:
            start = end  # failsafe

    return chunks


# --------------------------------------------------
# Timeout Wrapper
# --------------------------------------------------

def safe_make_chunks(
    text,
    max_chars=6000,
    overlap_chars=500,
    timeout=40
):
    """
    Runs chunking with timeout protection.
    If chunking exceeds timeout, skip safely.
    """

    with ThreadPoolExecutor(max_workers=1) as executor:

        future = executor.submit(
            make_chunks,
            text,
            max_chars,
            overlap_chars
        )

        try:
            return future.result(timeout=timeout)

        except TimeoutError:
            print("Chunking timed out (>40s). Skipping this source.")
            return []

        except Exception as e:
            print("Chunking error:", e)
            return []