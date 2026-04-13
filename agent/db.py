# agent/db.py
# Supabase persistence layer.
#
# Supabase table (run this SQL in Supabase → SQL Editor):
#
#   CREATE TABLE companies (
#     id            SERIAL PRIMARY KEY,
#     company_name  TEXT UNIQUE NOT NULL,
#     integrity_score FLOAT,
#     confidence_score FLOAT,
#     risk_label    TEXT,
#     metrics       JSONB,
#     report_text   TEXT,
#     ai_chunks     JSONB,
#     analyzed_at   TIMESTAMPTZ DEFAULT NOW()
#   );

import os
import json


def _get_client():
    """Build and return a Supabase client using secrets from env or st.secrets."""
    try:
        import streamlit as st
        url = st.secrets.get("SUPABASE_URL", "") or os.environ.get("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_KEY", "") or os.environ.get("SUPABASE_KEY", "")
    except Exception:
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")

    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set.")

    from supabase import create_client
    return create_client(url, key)


def get_company(company_name):
    """
    Look up a company by name.
    Returns a dict with keys: integrity_score, confidence_score, risk_label,
    metrics, report_text, ai_chunks — or None if not found.
    """
    try:
        client = _get_client()
        result = (
            client.table("companies")
            .select("*")
            .eq("company_name", company_name.lower().strip())
            .execute()
        )
        if result.data:
            row = result.data[0]
            # Supabase returns JSONB as dicts already, but guard against string
            for field in ("metrics", "ai_chunks"):
                if isinstance(row.get(field), str):
                    row[field] = json.loads(row[field])
            return row
    except Exception as e:
        print(f"[DB] get_company error: {e}")
    return None


def save_company(company_name, score_data, report_text, ai_chunks):
    """
    Upsert a company analysis into the database.
    Uses company_name as the unique key — re-analyzing overwrites the old record.
    """
    try:
        client = _get_client()
        data = {
            "company_name": company_name.lower().strip(),
            "integrity_score": score_data.get("integrity_score"),
            "confidence_score": score_data.get("confidence_score"),
            "risk_label": score_data.get("risk_label"),
            "metrics": score_data.get("metrics"),
            "report_text": report_text,
            "ai_chunks": ai_chunks,
        }
        client.table("companies").upsert(data, on_conflict="company_name").execute()
        return True
    except Exception as e:
        print(f"[DB] save_company error: {e}")
    return False


def list_companies():
    """
    Return a summary list of all analyzed companies, newest first.
    Each item has: company_name, integrity_score, risk_label, confidence_score, analyzed_at.
    """
    try:
        client = _get_client()
        result = (
            client.table("companies")
            .select("company_name, integrity_score, risk_label, confidence_score, analyzed_at")
            .order("analyzed_at", desc=True)
            .execute()
        )
        return result.data or []
    except Exception as e:
        print(f"[DB] list_companies error: {e}")
    return []