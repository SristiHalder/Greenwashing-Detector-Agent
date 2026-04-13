from dotenv import load_dotenv
import os

load_dotenv()


def _get(key):
    val = os.environ.get(key)
    if val:
        return val
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val:
            return val
    except Exception:
        pass
    raise ValueError(f"{key} is not set!")


# APIs
OPENAI_API_KEY = _get("OPENAI_API_KEY")
TAVILY_API_KEY = _get("TAVILY_API_KEY")

# Supabase
SUPABASE_URL = _get("SUPABASE_URL")
SUPABASE_KEY = _get("SUPABASE_KEY")
