import streamlit as st
import os

# Load .env file for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agent import db
from agent.pipeline import run_full_pipeline
from agent.explain_agent import answer_question
from agent.company_validator import company_validator  # ✅ NEW

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Greenwashing Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────

st.title("🌿 Greenwashing Detector")
st.caption(
    "AI-powered ESG inconsistency analysis - comparing what companies *say*, "
    "what they *report*, and what *others say* about them."
)
st.divider()

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Analyze a Company")

    company_input = st.text_input(
        "Company name",
        placeholder="e.g. Shell, H&M, Patagonia",
        key="company_input",
    )

    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    force_refresh = st.checkbox("🔄 Force re-analysis (ignore cached results)")

    st.divider()
    st.subheader("Previously Analyzed")

    try:
        companies = db.list_companies()
    except Exception:
        companies = []

    if companies:
        for c in companies:
            score = c.get("integrity_score")

            if score is None:
                badge = "⬜"
            elif score >= 80:
                badge = "🟢"
            elif score >= 60:
                badge = "🔵"
            elif score >= 40:
                badge = "🟡"
            else:
                badge = "🔴"

            btn_label = f"{badge} {c['company_name'].title()}  ({score}/100)" if score else c['company_name'].title()

            if st.button(btn_label, key=f"sidebar_{c['company_name']}", use_container_width=True):
                st.session_state["selected_company"] = c["company_name"]
                st.rerun()
    else:
        st.caption("No analyses yet.")

# ─────────────────────────────────────────────────────────────
# Score card
# ─────────────────────────────────────────────────────────────

LABEL_STYLES = {
    "High Integrity": {"bg": "#f0fdf4", "border": "#16a34a", "text": "#15803d"},
    "Moderate Integrity": {"bg": "#eff6ff", "border": "#2563eb", "text": "#1d4ed8"},
    "Weak Integrity": {"bg": "#fff7ed", "border": "#ea580c", "text": "#c2410c"},
    "Likely Greenwashing": {"bg": "#fef2f2", "border": "#dc2626", "text": "#b91c1c"},
}

def show_score_card(score_data):
    score = score_data.get("integrity_score")
    label = score_data.get("risk_label", "Unknown")
    confidence = score_data.get("confidence_score", 0)

    style = LABEL_STYLES.get(label, {"bg": "#f9fafb", "border": "#6b7280", "text": "#374151"})

    st.markdown(
        f"""
        <div style="
            background: {style['bg']};
            border-left: 6px solid {style['border']};
            border-radius: 8px;
            padding: 20px 28px;
            margin-bottom: 20px;
        ">
            <div style="font-size: 12px; font-weight: 600; color: #6b7280;">
                Integrity Score
            </div>
            <div style="font-size: 52px; font-weight: 800; color: {style['text']};">
                {score if score is not None else 'N/A'} / 100
            </div>
            <div style="font-size: 20px; font-weight: 700; color: {style['text']};">
                {label}
            </div>
            <div style="font-size: 13px; color: #6b7280;">
                Analysis confidence: <strong>{round(confidence, 2)} / 100</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────
# Results view
# ─────────────────────────────────────────────────────────────

def show_results(company, score_data, report_text, ai_chunks):
    st.subheader(f"Analysis: {company.title()}")
    show_score_card(score_data)

    tab1, tab2, tab3 = st.tabs(["📄 Forensic Report", "📊 Metrics", "💬 Ask Questions"])

    # ---------------- Report ----------------
    with tab1:
        if report_text:
            st.download_button(
                label="📥 Download Report",
                data=report_text,
                file_name=f"{company.lower().replace(' ', '_')}_esg_report.md",
                mime="text/markdown",
            )
            st.divider()
            st.markdown(report_text)
        else:
            st.warning("No report available.")

    # ---------------- Metrics ----------------
    with tab2:
        metrics = score_data.get("metrics", {})

        col1, col2, col3 = st.columns(3)

        def metric_card(title, value):
            st.markdown(
                f"""
                <div style="
                    background: #111827;
                    padding: 16px;
                    border-radius: 10px;
                    margin-bottom: 10px;
                    border: 1px solid #374151;
                ">
                    <div style="font-size: 12px; color: #9ca3af;">{title}</div>
                    <div style="font-size: 22px; font-weight: bold; color: #f9fafb;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col1:
            metric_card("Total Chunks", metrics.get("total_chunks"))
            metric_card("Claims", metrics.get("claims"))
            metric_card("Disclosures", metrics.get("disclosures"))

        with col2:
            metric_card("Transparency Ratio", round(metrics.get("transparency_ratio", 0), 3))
            metric_card("Controversy Ratio", round(metrics.get("controversy_ratio", 0), 3))
            metric_card("Marketing Gap", round(metrics.get("marketing_disclosure_gap", 0), 3))

        with col3:
            metric_card("AI Confidence", round(metrics.get("avg_ai_confidence", 0), 2))
            metric_card("Offset Ratio", round(metrics.get("offset_ratio", 0), 3))
            metric_card("Scope 3 Ratio", round(metrics.get("scope3_ratio", 0), 3))

    # ---------------- Ask Questions ----------------
    with tab3:
        question = st.text_input("Ask a question")
        if st.button("Ask"):
            answer = answer_question(
                company=company,
                user_question=question,
                messages=None,
                score=score_data,
                report=report_text,
                chunks=ai_chunks
            )
            st.write(answer)

# ─────────────────────────────────────────────────────────────
# Progress runner
# ─────────────────────────────────────────────────────────────

def run_with_progress(company):
    progress_bar = st.progress(0.0)
    stage_label = st.empty()

    stages = [
        ("Validating", 0.1),
        ("Collecting", 0.25),
        ("Extracting", 0.4),
        ("Chunking", 0.55),
        ("Running", 0.7),
        ("Computing", 0.85),
        ("Generating", 0.95),
    ]

    def on_status(msg):
        for label, pct in stages:
            if label.lower() in msg.lower():
                progress_bar.progress(pct)
                stage_label.markdown(f"**{label}...**")
                break

    result = run_full_pipeline(company, on_status=on_status)

    progress_bar.progress(1.0)
    stage_label.empty()
    progress_bar.empty()

    return result

# ─────────────────────────────────────────────────────────────
# Main logic
# ─────────────────────────────────────────────────────────────

if analyze_btn and company_input:
    active_company = company_input.strip()

    # ✅ VALIDATION STEP
    validation = company_validator(active_company)

    if validation["status"] == "invalid":
        st.error(f"❌ {validation['reason']}")
        st.stop()

    elif validation["status"] == "ambiguous":
        st.warning(f"⚠️ {validation['reason']}")
        st.info("Try a more specific name (e.g. 'Apple Inc' instead of 'Apple')")
        st.stop()

elif "selected_company" in st.session_state:
    active_company = st.session_state["selected_company"]
else:
    active_company = None

if not active_company:
    st.info("Enter a company name and click Analyze")
    st.stop()

try:
    cached = None if force_refresh else db.get_company(active_company)
except Exception:
    cached = None

if cached:
    score_data = {
        "integrity_score": cached["integrity_score"],
        "confidence_score": cached["confidence_score"],
        "risk_label": cached["risk_label"],
        "metrics": cached.get("metrics", {}),
    }

    show_results(active_company, score_data, cached.get("report_text", ""), cached.get("ai_chunks", []))

else:
    result = run_with_progress(active_company)

    if result["status"] == "success":
        db.save_company(
            active_company,
            result["score"],
            result["report"],
            result["ai_chunks"],
        )

        show_results(active_company, result["score"], result["report"], result["ai_chunks"])
