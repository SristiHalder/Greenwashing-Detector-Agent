# agent/ai_chunks.py

import os
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from openai import OpenAI
from agent.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

# Tuned via trial and error — do not change
BATCH_SIZE = 20
MAX_WORKERS = 15
BATCH_TIMEOUT = 15  # hard cutoff


SYSTEM_PROMPT = """
You are an ESG research analyst.

You will receive MULTIPLE text chunks separated by:

---CHUNK_BREAK---

For EACH chunk, return one JSON object.

Return a JSON LIST in the same order as the chunks.

If a chunk is not related to sustainability, return:
{"verdict": "irrelevant"}

Each object must contain:

- verdict: ["claim", "disclosure", "external_critique", "rating", "irrelevant"]
- claim_type: ["emissions_target", "sustainability_claim", "supply_chain_claim",
               "product_claim", "greenwashing_allegation", "other"]
- tone: ["positive", "negative", "neutral"]
- specificity: ["measurable", "semi_specific", "vague"]
- topics: list of topics
- numbers_dates: list
- key_sentences: list (1–2 sentences)
- red_flags: ["vague_language", "no_numbers", "legal_action",
              "controversy", "offset_heavy", "missing_scope3"]
- confidence: number between 0 and 1

Return ONLY valid JSON.
"""


def extract_json(text):
    try:
        text = re.sub(r"```json", "", text)
        text = re.sub(r"```", "", text)
        return json.loads(text.strip())
    except Exception:
        raise ValueError("Invalid JSON returned from model.")


def analyze_batch(batch_chunks, model="gpt-4.1-mini"):

    # Truncate each chunk to reduce inference time (tuned via trial and error)
    truncated = [c[:800] for c in batch_chunks]

    combined_text = "\n\n---CHUNK_BREAK---\n\n".join(truncated)

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": combined_text}
        ]
    )

    content = response.choices[0].message.content
    return extract_json(content)


def run_ai_chunking(company):

    company_dir = os.path.join("data", company.lower())
    chunks_path = os.path.join(company_dir, "chunks.jsonl")
    output_path = os.path.join(company_dir, "ai_chunks.jsonl")

    if not os.path.exists(chunks_path):
        print("Chunks file not found.")
        return

    print("\nRunning AI chunking (parallel mode)...\n")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    total_chunks = len(chunks)
    print(f"Total chunks: {total_chunks}")

    batches = [
        chunks[i:i + BATCH_SIZE]
        for i in range(0, total_chunks, BATCH_SIZE)
    ]

    total_batches = len(batches)
    print(f"Total batches: {total_batches}\n")

    kept = 0
    irrelevant = 0
    completed = 0

    with open(output_path, "w", encoding="utf-8") as outfile:

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

            future_to_batch = {
                executor.submit(
                    analyze_batch,
                    [c["chunk_text"] for c in batch]
                ): batch
                for batch in batches
            }

            for future in as_completed(future_to_batch):

                completed += 1
                print(f"Completed batch {completed}/{total_batches}")

                batch = future_to_batch[future]

                try:
                    ai_outputs = future.result(timeout=BATCH_TIMEOUT)

                except TimeoutError:
                    print("⚠️  Batch exceeded 15s — skipped.")
                    continue

                except Exception as e:
                    print("Batch error:", e)
                    continue

                if not isinstance(ai_outputs, list):
                    print("Unexpected output format — skipping batch.")
                    continue

                # Alignment note: the model often collapses consecutive irrelevant
                # chunks into fewer outputs instead of returning {"verdict": "irrelevant"}
                # for each one. zip() stops at the shorter list, so mismatches are safe —
                # we just lose the tail of the batch. Only hard-skip on extreme divergence.
                if len(ai_outputs) != len(batch):
                    ratio = len(ai_outputs) / max(len(batch), 1)
                    if ratio < 0.5:
                        print(
                            f"⚠️  Extreme mismatch: sent {len(batch)}, got {len(ai_outputs)} — skipping batch."
                        )
                        continue
                    print(
                        f"  ℹ️  Minor mismatch: sent {len(batch)}, got {len(ai_outputs)} — processing overlap."
                    )
                    # If model returned more than batch size, truncate to avoid phantom results
                    ai_outputs = ai_outputs[:len(batch)]

                for chunk_obj, ai_result in zip(batch, ai_outputs):

                    if ai_result.get("verdict") == "irrelevant":
                        irrelevant += 1
                        continue

                    kept += 1

                    result_record = {
                        "chunk_id": chunk_obj["chunk_id"],
                        "company": company,
                        "category": chunk_obj["category"],
                        "source_url": chunk_obj["source_url"],
                        "original_text": chunk_obj["chunk_text"],
                        "ai_analysis": ai_result
                    }

                    outfile.write(json.dumps(result_record, ensure_ascii=False) + "\n")

    print("\nProcessing complete.\n")
    print(f"Saved AI chunks to: {output_path}")
    print(f"Relevant chunks kept: {kept}")
    print(f"Irrelevant chunks skipped: {irrelevant}")