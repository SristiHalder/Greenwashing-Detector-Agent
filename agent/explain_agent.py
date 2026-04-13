# agent/explain_agent.py

import os
import json
from openai import OpenAI
from agent.config import OPENAI_API_KEY
from agent.utils import select_key_chunks  # shared utility

client = OpenAI(api_key=OPENAI_API_KEY)

REPORT_FILENAME = "report.md"


# --------------------------------------------------
# Load Context
# --------------------------------------------------

def load_context(company):

    company_dir = os.path.join("data", company.lower())

    score_path = os.path.join(company_dir, "score.json")
    report_path = os.path.join(company_dir, REPORT_FILENAME)
    chunks_path = os.path.join(company_dir, "ai_chunks.jsonl")

    if not os.path.exists(score_path):
        raise FileNotFoundError("score.json not found. Run scoring first.")

    if not os.path.exists(report_path):
        raise FileNotFoundError(f"{REPORT_FILENAME} not found. Run report generation first.")

    if not os.path.exists(chunks_path):
        raise FileNotFoundError("ai_chunks.jsonl not found. Run AI chunking first.")

    with open(score_path, "r", encoding="utf-8") as f:
        score = json.load(f)

    with open(report_path, "r", encoding="utf-8") as f:
        report = f.read()

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    return score, report, chunks


# --------------------------------------------------
# Main Conversational Function
# --------------------------------------------------

def answer_question(company, user_question, messages=None,
                    score=None, report=None, chunks=None):
    """
    Answer a user question grounded in the company's scoring data,
    forensic report, and most relevant evidence chunks.

    Args:
        company:       company name string
        user_question: the user's current question
        messages:      conversation history list (OpenAI message format).
                       Pass the same list across calls to maintain memory.
                       If None, starts a fresh single-turn conversation.
        score:         score dict (optional — if not provided, loads from file)
        report:        report text (optional — if not provided, loads from file)
        chunks:        ai_chunks list (optional — if not provided, loads from file)

    Returns:
        answer string
    """

    if score is None or chunks is None:
        score, report, chunks = load_context(company)
    elif report is None:
        report = ""

    # Query-aware chunk selection: re-rank chunks relevant to the question
    relevant_chunks = select_key_chunks(chunks, limit=8, query=user_question)

    context = {
        "risk_score": score["integrity_score"],
        "confidence_score": score["confidence_score"],
        "risk_label": score["risk_label"],
        "metrics": score["metrics"],
        "key_evidence": relevant_chunks
    }

    # Build the user turn for this question.
    # The forensic report summary is included so the agent can reference
    # narrative conclusions, not just raw metrics.
    report_summary = report[:2000] + "...[truncated]" if len(report) > 2000 else report

    user_turn = f"""
Scoring Data & Evidence:
{json.dumps(context, indent=2)}

Forensic Report Summary:
{report_summary}

Question:
{user_question}

Instructions:
- Then answer the question clearly and professionally.
- Reference specific claims and evidence.
- Cite source_url when relevant.
- If asked for sources, list them explicitly.
- Connect explanations to scoring metrics.
- Maintain transparency and traceability.
"""

    # Build message list
    if messages is None:
        # Single-turn mode: no history
        full_messages = [
            {
                "role": "system",
                "content": (
                    "You are an ESG forensic analyst. "
                    "Explain greenwashing assessment decisions transparently, "
                    "grounding all answers in the provided evidence and metrics."
                )
            },
            {"role": "user", "content": user_turn}
        ]
    else:
        # Multi-turn mode: append this question to existing conversation
        messages.append({"role": "user", "content": user_turn})
        full_messages = messages

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.2,
        messages=full_messages
    )

    answer = response.choices[0].message.content

    # Append assistant reply to history so next call has full context
    if messages is not None:
        messages.append({"role": "assistant", "content": answer})

    print("\n================ AI EXPLANATION ================\n")
    print(answer)
    print("\n================================================\n")

    return answer