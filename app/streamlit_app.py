"""
Khata Copilot — Streamlit UI (Phase 6).

Pure view over the existing pipeline: this file adds NO business logic. Every
number, flag, and answer on screen comes straight from core/agents functions
already built and tested in Phases 1-5 (orchestrator.process_page, core.db
queries, InsightsAgent, draft_reminders). Runs fully offline in mock mode
(KHATA_MOCK=1) so the demo never depends on a live Gemini/Groq key.

Theme adapted from the user's own Rag-Assistant-masterclass/zyro-rag-challenge
(dark glassmorphic cards, gradient hero title, source chips).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.insights_agent import InsightsAgent  # noqa: E402
from agents.reminder_agent import draft_reminders  # noqa: E402
from core import db  # noqa: E402
from core.config import mock_mode_enabled  # noqa: E402
from core.orchestrator import process_page  # noqa: E402
from core.schemas import InsightAnswer, PageResult, ReminderDraft, VerificationResult  # noqa: E402

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "sample_data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Khata Copilot",
    page_icon="📒",
    layout="wide",
    initial_sidebar_state="expanded",
)

SOURCE_LABELS = {
    "deterministic": ("🧮 Deterministic (SQL)", "chip-deterministic"),
    "llm": ("🤖 LLM", "chip-llm"),
    "extractive_fallback": ("📄 Extractive fallback", "chip-extractive_fallback"),
    "guardrail_refusal": ("🚫 Guardrail refusal", "chip-guardrail_refusal"),
    "empty": ("🕳️ No match", "chip-empty"),
    "template": ("📋 Template (no LLM)", "chip-template"),
}


# ── CSS (adapted from zyro-rag-challenge/app.py) ─────────────────────────────
def inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }

    .hero-header { text-align: center; padding: 20px 0 8px 0; }
    .hero-title {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero-subtitle { font-size: 1rem; color: #94a3b8; font-weight: 400; }

    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
        margin: 16px 0;
    }

    .stat-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.05));
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px; padding: 16px; text-align: center; margin: 8px 0;
    }
    .stat-number { font-size: 1.6rem; font-weight: 700; color: #a78bfa; word-break: break-word; }
    .stat-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

    .answer-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.05));
        border-left: 4px solid #6366f1; border-radius: 12px; padding: 18px 22px;
        margin: 12px 0; color: #e2e8f0; line-height: 1.6; font-size: 15px;
    }
    .blocked-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.08), rgba(220, 38, 38, 0.05));
        border-left: 4px solid #ef4444; border-radius: 12px; padding: 18px 22px;
        margin: 12px 0; color: #fca5a5; line-height: 1.6; font-size: 15px;
    }

    .entry-card {
        border-radius: 12px; padding: 14px 18px; margin: 8px 0;
        border-left: 4px solid transparent; background: rgba(17, 24, 39, 0.6);
    }
    .entry-ok {
        border-left-color: #22c55e;
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.06), rgba(17, 24, 39, 0.6));
    }
    .entry-flagged {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(17, 24, 39, 0.6));
    }

    .source-chip {
        display: inline-block; border-radius: 20px; padding: 3px 12px;
        margin: 2px 4px 2px 0; font-size: 12px; font-weight: 500;
        border: 1px solid transparent;
    }
    .chip-deterministic { background: rgba(34, 197, 94, 0.15); border-color: rgba(34, 197, 94, 0.35); color: #86efac; }
    .chip-llm { background: rgba(99, 102, 241, 0.15); border-color: rgba(99, 102, 241, 0.3); color: #a5b4fc; }
    .chip-extractive_fallback { background: rgba(245, 158, 11, 0.15); border-color: rgba(245, 158, 11, 0.35); color: #fcd34d; }
    .chip-template { background: rgba(245, 158, 11, 0.15); border-color: rgba(245, 158, 11, 0.35); color: #fcd34d; }
    .chip-guardrail_refusal { background: rgba(239, 68, 68, 0.15); border-color: rgba(239, 68, 68, 0.35); color: #fca5a5; }
    .chip-empty { background: rgba(148, 163, 184, 0.15); border-color: rgba(148, 163, 184, 0.3); color: #cbd5e1; }

    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0f1629, #0a0e1a) !important; border-right: 1px solid rgba(99, 102, 241, 0.1); }
    [data-testid="stSidebar"] .stMarkdown h1, [data-testid="stSidebar"] .stMarkdown h2, [data-testid="stSidebar"] .stMarkdown h3 { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] .stMarkdown li { color: #cbd5e1 !important; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# ── Cached singletons (shared across tabs/reruns) ────────────────────────────
@st.cache_resource(show_spinner=False)
def get_conn():
    # Streamlit reruns the script in a new thread per session/interaction, but
    # this single cached connection is shared across all of them — sqlite3
    # needs check_same_thread=False for that (db.connect() defaults to True).
    import sqlite3

    from core.config import DB_PATH

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    db.init_db(conn)
    return conn


@st.cache_resource(show_spinner=False)
def get_agent(_conn) -> InsightsAgent:
    return InsightsAgent(_conn)


def refresh_insights() -> None:
    get_agent(get_conn()).refresh_index()


# ── Small render helpers ──────────────────────────────────────────────────────
def source_chip(source: str) -> str:
    label, cls = SOURCE_LABELS.get(source, (source, "chip-empty"))
    return f'<span class="source-chip {cls}">{label}</span>'


def stat_card(value: str, label: str) -> None:
    st.markdown(
        f'<div class="stat-card"><div class="stat-number">{value}</div>'
        f'<div class="stat-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def render_entry_card(
    name: str, amount: float, date, status: str, confidence: float,
    flagged: bool, messages=None,
) -> None:
    cls = "entry-flagged" if flagged else "entry-ok"
    badge = "🚩 NEEDS REVIEW" if flagged else "✅ OK"
    badge_color = "#f87171" if flagged else "#4ade80"
    date_label = date or "—"
    msg_html = ""
    if messages:
        msg_html = (
            "<div style='margin-top:8px; color:#fca5a5; font-size:13px;'>"
            + "<br>".join(messages) + "</div>"
        )
    st.markdown(f"""
    <div class="entry-card {cls}">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-weight:600; color:#e2e8f0; font-size:15px;">{name or "UNREADABLE"}</div>
            <div style="font-size:12px; color:{badge_color};">{badge}</div>
        </div>
        <div style="color:#94a3b8; font-size:13px; margin-top:4px;">
            ₹{amount:,.2f} &nbsp;•&nbsp; {status} &nbsp;•&nbsp; {date_label} &nbsp;•&nbsp; confidence {confidence:.0%}
        </div>
        {msg_html}
    </div>
    """, unsafe_allow_html=True)


def _flagged_map(verification: VerificationResult):
    page_level = any(
        i.severity in ("warning", "error") and i.entry_index == -1 for i in verification.issues
    )
    per_entry: dict[int, list[str]] = {}
    for i in verification.issues:
        if i.severity in ("warning", "error") and i.entry_index >= 0:
            per_entry.setdefault(i.entry_index, []).append(i.message)
    return page_level, per_entry


def render_scan_result(result: PageResult) -> None:
    extraction = result.extraction
    verification = result.verification

    if mock_mode_enabled() or extraction.degraded:
        st.warning("⚙️ MOCK MODE — this extraction is synthetic demo data (no live Gemini call was made).")

    for err in result.stage_errors:
        st.error(err)

    c1, c2, c3 = st.columns(3)
    with c1:
        stat_card(f"₹{verification.computed_total:,.2f}", "Computed Total")
    with c2:
        written = f"₹{verification.written_total:,.2f}" if verification.written_total is not None else "—"
        stat_card(written, "Written Total")
    with c3:
        verdict_label = "✅ ACCEPT" if verification.verdict == "accept" else "🚩 NEEDS REVIEW"
        stat_card(verdict_label, "Verdict")

    if verification.total_difference is not None and abs(verification.total_difference) > 0.005:
        st.markdown(
            f"<div style='color:#fbbf24; font-size:13px; margin:4px 0 12px 0;'>"
            f"Difference (written − computed): ₹{verification.total_difference:,.2f}</div>",
            unsafe_allow_html=True,
        )

    page_level, per_entry = _flagged_map(verification)
    if page_level:
        page_msgs = [i.message for i in verification.issues if i.entry_index == -1]
        st.markdown(
            f'<div class="blocked-card">{"<br>".join(page_msgs)}</div>', unsafe_allow_html=True,
        )

    st.markdown("#### Extracted Entries")
    if not extraction.entries:
        st.info("No entries were extracted from this page.")
    for idx, entry in enumerate(extraction.entries):
        flagged = page_level or idx in per_entry
        render_entry_card(
            entry.name, entry.amount, entry.date, entry.status, entry.confidence,
            flagged, per_entry.get(idx),
        )

    if result.ingest:
        st.caption(
            f"Ledger: {result.ingest.inserted} inserted, {result.ingest.updated} updated "
            f"(page verdict: {result.ingest.page_verdict})."
        )
    else:
        st.error("Ledger ingest did not run for this page.")


def render_insight_answer(ans: InsightAnswer) -> None:
    card_cls = "blocked-card" if ans.is_blocked else "answer-card"
    degraded_note = (
        " <span style='color:#fbbf24;font-size:11px;'>(degraded — no LLM)</span>" if ans.degraded else ""
    )
    st.markdown(f"""
    <div class="{card_cls}">
        <div style="color:#a5b4fc; font-weight:600; font-size:13px; margin-bottom:6px;">🧑 {ans.question}</div>
        <div style="margin-bottom:10px;">{ans.answer}</div>
        <div>{source_chip(ans.source)}{degraded_note}</div>
    </div>
    """, unsafe_allow_html=True)
    if ans.citations:
        noun = "entry" if len(ans.citations) == 1 else "entries"
        with st.expander(f"📚 {len(ans.citations)} source {noun}"):
            for c in ans.citations:
                st.code(c, language=None)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


def render_reminder_card(d: ReminderDraft) -> None:
    since = f" (since {d.since_date})" if d.since_date else ""
    st.markdown(f"""
    <div class="answer-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-weight:600; color:#e2e8f0; font-size:15px;">{d.customer_name}</div>
            <div>{source_chip(d.source)}</div>
        </div>
        <div style="color:#94a3b8; font-size:13px; margin:4px 0 10px 0;">
            Outstanding: ₹{d.amount:,.2f}{since}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.text_area(
        "Reminder message (copy-paste ready)", value=d.message, height=110,
        key=f"reminder_msg_{d.customer_name}", label_visibility="collapsed",
    )
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# ── App ───────────────────────────────────────────────────────────────────────
inject_css()

with st.sidebar:
    st.markdown("### 📒 Khata Copilot")
    st.markdown("**Agentic khata digitizer**")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    if mock_mode_enabled():
        st.markdown(
            '<span class="source-chip chip-template">⚙️ MOCK MODE — offline demo data</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="source-chip chip-deterministic">🟢 Live mode</span>', unsafe_allow_html=True,
        )
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Demo")
    if st.button("🎬 Load Demo Data", help="Clears the ledger and loads a fixed, known-good set of entries."):
        db.seed_demo_data(get_conn())
        refresh_insights()
        st.session_state.pop("last_scan", None)
        st.session_state.pop("reminders", None)
        st.session_state["chat_history"] = []
        st.success("Demo data loaded.")
        st.rerun()
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### How it works")
    st.markdown(
        "- **Scan**: photo → Vision → Verification → Ledger\n"
        "- **Ledger**: search + balances + flagged review\n"
        "- **Insights**: deterministic stats + grounded Q&A\n"
        "- **Reminders**: bilingual drafts, copy-only"
    )
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center; color:#64748b; font-size:12px;'>"
        "Vision → Verification → Ledger → Insights → Reminders<br>"
        "NIAT TakeOver'26"
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("""
<div class="hero-header">
    <div class="hero-title">📒 Khata Copilot</div>
    <div class="hero-subtitle">Honest confidence scores, self-flagging entries, zero hallucinated numbers</div>
</div>
<div class="custom-divider"></div>
""", unsafe_allow_html=True)

conn = get_conn()
tab_scan, tab_ledger, tab_insights, tab_reminders = st.tabs(
    ["📷 Scan", "📒 Ledger", "💡 Insights", "🔔 Reminders"]
)

with tab_scan:
    st.markdown("#### Scan a Khata Page")
    uploaded = st.file_uploader("Upload a photo of the khata page", type=["jpg", "jpeg", "png"])
    scan_clicked = st.button("🔍 Scan Page", type="primary", disabled=uploaded is None)

    if scan_clicked and uploaded is not None:
        content = uploaded.getvalue()
        # Hash the bytes (not a random UUID) so re-uploading the SAME photo in
        # mock mode reuses the SAME source_image — the db's ON CONFLICT dedup
        # then updates the existing rows instead of piling up duplicates.
        content_hash = hashlib.sha256(content).hexdigest()[:16]
        suffix = Path(uploaded.name).suffix or ".jpg"
        dest = UPLOAD_DIR / f"{content_hash}{suffix}"
        if not dest.exists():
            dest.write_bytes(content)
        with st.spinner("Running Vision → Verification → Ledger..."):
            st.session_state["last_scan"] = process_page(str(dest), conn)
            st.session_state["last_scan_image"] = dest
        refresh_insights()

    last_scan = st.session_state.get("last_scan")
    if last_scan is not None:
        preview_col, result_col = st.columns([1, 2])
        with preview_col:
            last_scan_image = st.session_state.get("last_scan_image")
            if last_scan_image is not None and Path(last_scan_image).exists():
                st.image(str(last_scan_image), caption="Scanned page", use_container_width=True)
        with result_col:
            render_scan_result(last_scan)
    else:
        st.info("Upload a khata photo and click Scan to see extraction results here.")

with tab_ledger:
    st.markdown("#### All Ledger Entries")
    entries = db.get_all_entries(conn)
    search = st.text_input("Search by customer name", "", placeholder="e.g. Ramesh")
    filtered = (
        [e for e in entries if search.strip().lower() in e.customer_name.lower()]
        if search.strip() else entries
    )
    if filtered:
        df = pd.DataFrame([e.model_dump() for e in filtered])
        df["needs_review"] = df["needs_review"].map({True: "🚩", False: ""})
        df = df[[
            "customer_name", "amount", "raw_date", "status", "confidence",
            "needs_review", "source_image", "scanned_at",
        ]].rename(columns={
            "customer_name": "Customer", "amount": "Amount", "raw_date": "Date",
            "status": "Status", "confidence": "Confidence", "needs_review": "Flagged",
            "source_image": "Source Image", "scanned_at": "Scanned At",
        })
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No entries in the ledger yet — scan a page in the Scan tab.")

    st.markdown("#### Customer Balances")
    balances = db.get_all_balances(conn)
    if balances:
        bal_df = pd.DataFrame([b.model_dump() for b in balances]).rename(columns={
            "name": "Customer", "unpaid_total": "Outstanding (₹)",
            "paid_total": "Paid (₹)", "entry_count": "Entries",
        })
        st.dataframe(bal_df, use_container_width=True, hide_index=True)
    else:
        st.info("No outstanding balances.")

    st.markdown("#### 🚩 Needs Review")
    review = db.get_entries_needing_review(conn)
    if not review:
        st.success("Nothing flagged — every stored entry passed verification.")
    else:
        for e in review:
            render_entry_card(e.customer_name, e.amount, e.raw_date, e.status, e.confidence, flagged=True)

with tab_insights:
    st.markdown("#### At a Glance")
    agent = get_agent(conn)
    stats = agent.stats()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card(f"₹{stats.total_outstanding:,.0f}", "Total Outstanding")
    with c2:
        stat_card(str(stats.customer_count), "Customers Owing")
    with c3:
        stat_card(str(stats.flagged_count), "Flagged Entries")
    with c4:
        top = stats.top_defaulters[0].name if stats.top_defaulters else "—"
        stat_card(top, "Top Defaulter")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 💬 Ask Your Ledger")
    question = st.text_input(
        "Ask a question", placeholder="e.g. Who owes the most? / Kitna baki hai?",
        label_visibility="collapsed", key="insights_q",
    )
    ask_clicked = st.button("Ask", type="primary", key="insights_ask")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if ask_clicked and question.strip():
        with st.spinner("Thinking..."):
            answer = agent.ask(question.strip())
        st.session_state.chat_history.insert(0, answer)

    for ans in st.session_state.chat_history:
        render_insight_answer(ans)

with tab_reminders:
    st.markdown("#### Payment Reminders")
    if st.button("🔄 Draft Reminders", type="primary"):
        with st.spinner("Drafting bilingual reminders..."):
            st.session_state["reminders"] = draft_reminders(conn)

    drafts = st.session_state.get("reminders")
    if drafts is None:
        st.info("Click 'Draft Reminders' to generate messages for every customer with an outstanding balance.")
    elif not drafts:
        st.success("No outstanding balances — nothing to remind anyone about.")
    else:
        for d in drafts:
            render_reminder_card(d)
