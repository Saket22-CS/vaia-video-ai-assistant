import streamlit as st
import time
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VAIA · Video AI Assistant",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=Outfit:wght@300;400;500;600;700&display=swap');

/* ══════════════════════════════════════════
   TOKENS
══════════════════════════════════════════ */
:root {
    --ink:        #0d0d12;
    --ink-2:      #16161f;
    --ink-3:      #1e1e2b;
    --ink-4:      #272736;
    --line:       #2e2e42;
    --line-soft:  #232335;
    --muted:      #5a5a7a;
    --text:       #d8d8e8;
    --text-soft:  #9898b8;
    --gold:       #e8c97a;
    --gold-dim:   #c8a950;
    --teal:       #5de0c8;
    --teal-dim:   #3db8a0;
    --rose:       #f07090;
    --white:      #f0f0f8;

    --r-sm:  6px;
    --r-md:  12px;
    --r-lg:  18px;
    --r-xl:  24px;
}

/* ══════════════════════════════════════════
   GLOBAL RESET
══════════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background: var(--ink) !important;
    color: var(--text) !important;
}

.stApp { background: var(--ink) !important; }

/* Subtle noise texture overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.6;
}

/* Ambient glow */
.stApp::after {
    content: '';
    position: fixed;
    top: -200px;
    left: 50%;
    transform: translateX(-50%);
    width: 800px;
    height: 500px;
    background: radial-gradient(ellipse at center, rgba(93,224,200,0.04) 0%, rgba(232,201,122,0.03) 40%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}
            
/* ══════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════ */

[data-testid="stSidebar"] {
    background: #0e1117 !important;
    border-right: 1px solid #1f2937 !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

/* Sidebar inputs */
[data-testid="stSidebar"] .stTextInput > div > div > input,
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #161b22 !important;
    border: 1px solid #2d3748 !important;
    border-radius: var(--r-sm) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
}                 

/* ══════════════════════════════════════════
   WORDMARK
══════════════════════════════════════════ */
.wordmark {
    display: flex;
    align-items: baseline;
    gap: 0.3rem;
    margin-bottom: 0.25rem;
}

.wordmark-v {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    line-height: 1;
    color: var(--gold);
}

.wordmark-rest {
    font-family: 'Outfit', sans-serif;
    font-size: 1.1rem;
    font-weight: 300;
    letter-spacing: 0.25em;
    color: var(--text-soft);
    text-transform: uppercase;
}

.sidebar-tagline {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

/* ══════════════════════════════════════════
   PAGE HERO
══════════════════════════════════════════ */
.page-hero {
    padding: 2.5rem 0 2rem;
    margin-bottom: 0.5rem;
}

.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--teal);
    letter-spacing: 0.3em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

.hero-heading {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.4rem, 4vw, 3.6rem);
    line-height: 1.05;
    color: var(--white);
    margin: 0 0 0.8rem;
}

.hero-heading em {
    font-style: italic;
    color: var(--gold);
}

.hero-desc {
    font-family: 'Outfit', sans-serif;
    font-size: 0.95rem;
    color: var(--text-soft);
    font-weight: 300;
    line-height: 1.7;
    max-width: 520px;
}

/* ══════════════════════════════════════════
   HORIZONTAL RULE
══════════════════════════════════════════ */
.hr-fancy {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 1.5rem 0;
}

.hr-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--line), transparent);
}

.hr-diamond {
    color: var(--gold-dim);
    font-size: 0.5rem;
}

/* ══════════════════════════════════════════
   CAPS LABEL
══════════════════════════════════════════ */
.caps-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.caps-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--line-soft);
}

/* ══════════════════════════════════════════
   CARDS
══════════════════════════════════════════ */
.card {
    background: #11161d
    border: 1px solid #2d3748;
    border-radius: var(--r-lg);
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s, transform 0.2s;
    margin-bottom: 1rem;
}

.card:hover {
    border-color: var(--line-soft);
    transform: translateY(-1px);
}

.card-accent-gold::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
}

.card-accent-teal::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--teal), transparent);
}

.card-accent-rose::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--rose), transparent);
}

.card-title {
    font-family: 'Outfit', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    width: 100%;       /* ← add this */
}

.card-body {
    font-size: 0.875rem;
    line-height: 1.8;
    color: var(--text-soft);
    font-weight: 300;
}
            
/* Mobile: stack intelligence columns */
@media (max-width: 768px) {
    [data-testid="column"] {
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }

    .hero-heading {
        font-size: 2rem !important;
    }

    .title-banner-text {
        font-size: 1.2rem !important;
    }

    .chat-wrap {
        max-height: 340px !important;
    }
}

/* Title banner */
.title-banner {
    background: linear-gradient(135deg, var(--ink-3) 0%, var(--ink-4) 100%);
    border: 1px solid #2d3748;
    border-radius: var(--r-lg);
    padding: 1.75rem 2rem;
    position: relative;
    overflow: hidden;
    margin-bottom: 1.5rem;
}

.title-banner::before {
    content: '◈';
    position: absolute;
    right: 1.5rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 4rem;
    color: var(--line);
    pointer-events: none;
    user-select: none;
}

.title-banner-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 0.5rem;
}

.title-banner-text {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: var(--white);
    line-height: 1.2;
}

/* ══════════════════════════════════════════
   TRANSCRIPT
══════════════════════════════════════════ */
.transcript-wrap {
    background: #161b22;
    border: 1px solid var(--line-soft);
    border-radius: var(--r-md);
    padding: 1.25rem;
    max-height: 320px;
    overflow-y: auto;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    line-height: 2;
    color: #b8c1cc;
    white-space: pre-wrap;
    word-break: break-word;
}

/* ══════════════════════════════════════════
   PIPELINE STEPS
══════════════════════════════════════════ */
.step-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0.9rem;
    border-radius: var(--r-sm);
    margin-bottom: 0.35rem;
    background: #161b22;
    border: 1px solid var(--line-soft);
    font-size: 0.78rem;
    font-family: 'Outfit', sans-serif;
    font-weight: 400;
    color: var(--text-soft);
    transition: background 0.2s;
}

.step-row.done {
    border-color: rgba(93,224,200,0.2);
    color: var(--text);
}

.step-row.active {
    border-color: rgba(232,201,122,0.3);
    background: rgba(232,201,122,0.04);
    color: var(--gold);
}

.step-icon {
    width: 7px; height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}

.step-icon.pending { background: var(--line); }
.step-icon.active  {
    background: var(--gold);
    box-shadow: 0 0 8px var(--gold);
    animation: glow-pulse 1.4s ease-in-out infinite;
}
.step-icon.done    { background: var(--teal); }

@keyframes glow-pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px var(--gold); }
    50%       { opacity: 0.5; box-shadow: 0 0 3px var(--gold); }
}

/* ══════════════════════════════════════════
   BUTTONS
══════════════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, #e8c97a 0%, #c8a950 100%) !important;
    color: #0d0d12 !important;
    border: none !important;
    border-radius: var(--r-sm) !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(232,201,122,0.3) !important;
    filter: brightness(1.07) !important;
}

.stButton > button[kind="secondary"] {
    background: #161b22 !important;
    color: var(--muted) !important;
    border: 1px solid #2d3748 !important;
}

/* ══════════════════════════════════════════
   CHAT
══════════════════════════════════════════ */
.chat-wrap {
    background: #11161d;
    border: 1px solid #2d3748;
    border-radius: var(--r-lg);
    padding: 1.25rem;
    max-height: 460px;
    overflow-y: auto;
    margin-bottom: 1rem;
    scroll-behavior: smooth;
}

.chat-msg { margin-bottom: 1.1rem; }

.chat-label-user {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--gold-dim);
    margin-bottom: 0.3rem;
    text-align: right;
}

.chat-label-bot {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--teal-dim);
    margin-bottom: 0.3rem;
}

.bubble-user {
    background: rgba(232,201,122,0.08);
    border: 1px solid rgba(232,201,122,0.18);
    border-radius: var(--r-md) var(--r-sm) var(--r-sm) var(--r-md);
    padding: 0.7rem 1rem;
    font-size: 0.85rem;
    line-height: 1.65;
    color: var(--text);
    max-width: 88%;
    margin-left: auto;
    text-align: right;
}

.bubble-bot {
    background: rgba(93,224,200,0.05);
    border: 1px solid rgba(93,224,200,0.15);
    border-radius: var(--r-sm) var(--r-md) var(--r-md) var(--r-sm);
    padding: 0.7rem 1rem;
    font-size: 0.85rem;
    line-height: 1.65;
    color: var(--text);
    max-width: 88%;
}

/* ══════════════════════════════════════════
   EMPTY STATE
══════════════════════════════════════════ */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 5rem 2rem 4rem;
    text-align: center;
}

.empty-glyph {
    font-family: 'DM Serif Display', serif;
    font-size: 5rem;
    color: var(--line);
    line-height: 1;
    margin-bottom: 1.5rem;
    letter-spacing: -0.05em;
}

.empty-heading {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: var(--text-soft);
    margin-bottom: 0.75rem;
}

.empty-sub {
    font-size: 0.85rem;
    color: var(--muted);
    line-height: 1.8;
    max-width: 360px;
    font-weight: 300;
}

/* ══════════════════════════════════════════
   PILLS / TAGS
══════════════════════════════════════════ */
.pill {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-right: 0.4rem;
}

.pill-gold  { background: rgba(232,201,122,0.12); color: var(--gold); border: 1px solid rgba(232,201,122,0.2); }
.pill-teal  { background: rgba(93,224,200,0.1);   color: var(--teal); border: 1px solid rgba(93,224,200,0.2); }
.pill-rose  { background: rgba(240,112,144,0.1);  color: var(--rose); border: 1px solid rgba(240,112,144,0.2); }

/* ══════════════════════════════════════════
   MISC OVERRIDES
══════════════════════════════════════════ */
.stProgress > div > div > div { background: var(--gold) !important; }
.stSpinner > div { border-top-color: var(--gold) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
label { color: var(--muted) !important; }

hr {
    border: none !important;
    border-top: 1px solid var(--line) !important;
    margin: 1.5rem 0 !important;
}

.stExpander {
    background: #11161d !important;
    border: 1px solid #2d3748 !important;
    border-radius: var(--r-md) !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--line); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* Streamlit radio & select cleanup */
.stSelectbox [data-baseweb="select"] {
    background: #161b22 !important;
    border-color: #2d3748 !important;
}
            
/* Download button matches theme */
[data-testid="stDownloadButton"] > button {
    background: #161b22 !important;
    color: var(--teal) !important;
    border: 1px solid rgba(93,224,200,0.25) !important;
    border-radius: var(--r-sm) !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s !important;
}

[data-testid="stDownloadButton"] > button:hover {
    border-color: var(--teal) !important;
    box-shadow: 0 4px 16px rgba(93,224,200,0.15) !important;
    transform: translateY(-1px) !important;
}     
            
[data-testid="stSidebar"] {
    box-shadow: inset -1px 0 0 #1f2937;
}
            
/* Smooth UI transitions */
* {
    transition:
        background-color 0.18s ease,
        border-color 0.18s ease,
        color 0.18s ease;
}                             
            
</style>
""", unsafe_allow_html=True)

# ─── Session State ──────────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "pipeline_steps": {},
    "pipeline_done": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ────────────────────────────────────────────────────────────────────
def set_step(key, state):
    st.session_state.pipeline_steps[key] = state

def render_step(icon, label, key):
    s = st.session_state.pipeline_steps.get(key, "pending")
    row_class = f"step-row {s}"
    icon_class = f"step-icon {s}"
    st.markdown(f"""
    <div class="{row_class}">
        <div class="{icon_class}"></div>
        <span>{icon}&nbsp; {label}</span>
    </div>""", unsafe_allow_html=True)

def hr():
    st.markdown("""
    <div class="hr-fancy">
        <div class="hr-line"></div>
        <div class="hr-diamond">◆</div>
        <div class="hr-line"></div>
    </div>""", unsafe_allow_html=True)

def format_llm_text(text: str) -> str:
    """Convert plain LLM output into clean HTML list items."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    items = ""

    for line in lines:
        # strip leading bullets/dashes/numbers the LLM may have added
        clean = line.lstrip("-•*·1234567890.) ").strip()

        if clean:
            items += (
                f'<li style="margin-bottom:0.5rem;padding-left:0.25rem">'
                f"{clean}</li>"
            )

    if items:
        return (
            '<ul style="padding-left:1rem;margin:0;list-style:none">'
            f"{items}</ul>"
        )

    return f'<p style="margin:0">{text}</p>'


def build_export(r: dict) -> str:
    """Generate downloadable meeting report text."""

    lines = []

    lines.append("=" * 60)
    lines.append("  VAIA — Meeting Export")
    lines.append("=" * 60)

    lines.append(f"\nTITLE\n{'─' * 40}\n{r['title']}\n")

    lines.append(f"SUMMARY\n{'─' * 40}\n{r['summary']}\n")

    lines.append(
        f"ACTION ITEMS\n{'─' * 40}\n{r['action_items']}\n"
    )

    lines.append(
        f"KEY DECISIONS\n{'─' * 40}\n{r['key_decisions']}\n"
    )

    lines.append(
        f"OPEN QUESTIONS\n{'─' * 40}\n{r['open_questions']}\n"
    )

    lines.append(
        f"FULL TRANSCRIPT\n{'─' * 40}\n{r['transcript']}\n"
    )

    return "\n".join(lines)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="wordmark">
        <span class="wordmark-v">V</span>
        <span class="wordmark-rest">AIA</span>
    </div>
    <div class="sidebar-tagline">Video Intelligence System</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="caps-label">Source</div>', unsafe_allow_html=True)
    source = st.text_input(
        "source",
        placeholder="YouTube URL or /path/to/file.mp4",
        label_visibility="collapsed"
    )

    st.markdown('<div class="caps-label" style="margin-top:1rem">Language</div>', unsafe_allow_html=True)
    language = st.selectbox("Language", ["english", "hinglish"], label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("◈  Run Analysis", use_container_width=True)
    if st.session_state.pipeline_done:
        if st.button("↺  New Session", use_container_width=True, type="secondary"):
            st.session_state.result = None
            st.session_state.chat_history = []
            st.session_state.pipeline_steps = {}
            st.session_state.pipeline_done = False
            st.session_state["chat_input_val"] = ""
            st.rerun()

    if st.session_state.pipeline_done or st.session_state.pipeline_steps:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="caps-label">Pipeline</div>', unsafe_allow_html=True)
        steps = [
            ("▸", "Audio Processing",  "audio"),
            ("▸", "Transcription",     "transcript"),
            ("▸", "Title Generation",  "title"),
            ("▸", "Summarisation",     "summary"),
            ("▸", "Extraction",        "extract"),
            ("▸", "RAG Engine",        "rag"),
        ]
        for icon, label, key in steps:
            render_step(icon, label, key)

    # Model info at bottom
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="border-top:1px solid var(--line-soft);padding-top:1rem;margin-top:auto">
        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:var(--muted);letter-spacing:0.1em;line-height:2">
            LLM · mistral-small<br>
            STT · whisper / sarvam<br>
            VDB · chromadb (local)<br>
            EMB · all-MiniLM-L6-v2
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Main ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-hero">
    <div class="hero-eyebrow">◈ AI-Powered Meeting Intelligence</div>
    <h1 class="hero-heading">Understand every<br><em>word spoken.</em></h1>
    <p class="hero-desc">
        Paste a YouTube link or drop a local video file.
        VAIA transcribes, summarises, extracts insights,
        and lets you converse with the content.
    </p>
</div>
""", unsafe_allow_html=True)

# Capability pills
st.markdown("""
<div style="margin-bottom:2rem">
    <span class="pill pill-gold">Transcription</span>
    <span class="pill pill-teal">Summarisation</span>
    <span class="pill pill-rose">Extraction</span>
    <span class="pill pill-gold">RAG Chat</span>
    <span class="pill pill-teal">Hindi / Hinglish</span>
</div>
""", unsafe_allow_html=True)

hr()

# ─── Pipeline Execution ─────────────────────────────────────────────────────────
if run_btn:
    if not source.strip():
        st.error("Please provide a YouTube URL or local file path.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}

        progress_ph = st.empty()

        try:
            with progress_ph.container():
                st.markdown("""
                <div class="card card-accent-gold" style="text-align:center;padding:1.5rem">
                    <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:var(--gold);letter-spacing:0.2em;text-transform:uppercase">
                        ◈ &nbsp; Pipeline Running — Monitor progress in sidebar
                    </div>
                </div>""", unsafe_allow_html=True)

            set_step("audio", "active")
            chunks = process_input(source)
            set_step("audio", "done")

            set_step("transcript", "active")
            transcript = transcribe_all(chunks, language)
            set_step("transcript", "done")

            set_step("title", "active")
            title = generate_title(transcript)
            set_step("title", "done")

            set_step("summary", "active")
            summary = summarize(transcript)
            set_step("summary", "done")

            set_step("extract", "active")
            action_items = extract_action_items(transcript)
            decisions    = extract_key_decisions(transcript)
            questions    = extract_questions(transcript)
            set_step("extract", "done")

            set_step("rag", "active")
            rag_chain = build_rag_chain(transcript)
            set_step("rag", "done")

            st.session_state.result = {
                "title":         title,
                "transcript":    transcript,
                "summary":       summary,
                "action_items":  action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain":     rag_chain,
            }
            st.session_state.pipeline_done = True
            progress_ph.markdown("""
            <div class="card card-accent-teal" style="text-align:center;padding:1.5rem">
                <div style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:var(--teal);margin-bottom:0.3rem">
                    ✓ &nbsp; Analysis Complete
                </div>
                <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:var(--muted);letter-spacing:0.15em;text-transform:uppercase">
                    Transcript · Summary · Extraction · RAG — all ready
                </div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1.5)
            progress_ph.empty()
            st.rerun()

        except Exception as e:
            for k in ["audio","transcript","title","summary","extract","rag"]:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            progress_ph.error(f"Pipeline error: {e}")

# ─── Results ────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    # ── Title banner
    st.markdown(f"""
    <div class="title-banner">
        <div class="title-banner-label">◈ Session Title</div>
        <div class="title-banner-text">{r['title']}</div>
    </div>""", unsafe_allow_html=True)

    # ── Summary + Transcript
    col_sum, col_tr = st.columns([3, 2], gap="large")


    dl_col, _ = st.columns([2, 5])
    with dl_col:
        export_text = build_export(r)
        st.download_button(
            label="↓  Export Report",
            data=export_text,
            file_name=f"{r['title'][:40].replace(' ','_')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col_sum:
        summary_words = len(r['summary'].split())
        transcript_words = len(r['transcript'].split())
        compression = round((1 - summary_words / max(transcript_words, 1)) * 100)
        st.markdown(f"""
        <div class="card card-accent-gold">
            <div class="card-title">
                ◧ Summary
                <span style="margin-left:auto;display:flex;gap:0.4rem">
                    <span class="pill pill-gold">{summary_words} words</span>
                    <span class="pill pill-teal">{compression}% compressed</span>
                </span>
            </div>
            <div class="card-body">{r['summary']}</div>
        </div>""", unsafe_allow_html=True)

    with col_tr:
        st.markdown('<div class="caps-label">Full Transcript</div>', unsafe_allow_html=True)
        word_count = len(r["transcript"].split())
        char_count = len(r["transcript"])
        read_mins  = max(1, round(word_count / 200))
        with st.expander(f"Expand to read · {word_count:,} words · ~{read_mins} min read", expanded=False):
            st.markdown(f'<div class="transcript-wrap">{r["transcript"]}</div>', unsafe_allow_html=True)

    hr()

    # ── Intelligence row
    st.markdown('<div class="caps-label">Meeting Intelligence</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(f"""
        <div class="card card-accent-teal">
            <div class="card-title" style="color:var(--teal)">✓ Action Items</div>
            <div class="card-body">{format_llm_text(r['action_items'])}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card card-accent-gold">
            <div class="card-title" style="color:var(--gold)">◈ Key Decisions</div>
            <div class="card-body">{format_llm_text(r['key_decisions'])}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card card-accent-rose">
            <div class="card-title" style="color:var(--rose)">? Open Questions</div>
            <div class="card-body">{format_llm_text(r['open_questions'])}</div>
        </div>""", unsafe_allow_html=True)

    hr()

    # ── RAG Chat
    st.markdown("""
    <div style="margin-bottom:1.25rem">
        <div class="hero-eyebrow" style="margin-bottom:0.4rem">◈ Conversational Interface</div>
        <div style="font-family:'DM Serif Display',serif;font-size:1.4rem;color:var(--white)">
            Chat with your meeting
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Chat history
    if st.session_state.chat_history:
        chat_html = '<div class="chat-wrap">'
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_html += f"""
                <div class="chat-msg">
                    <div class="chat-label-user">You</div>
                    <div class="bubble-user">{msg['content']}</div>
                </div>"""
            else:
                chat_html += f"""
                <div class="chat-msg">
                    <div class="chat-label-bot">VAIA</div>
                    <div class="bubble-bot">{msg['content']}</div>
                </div>"""
        chat_html += '<div id="chat-bottom"></div></div>'
        st.markdown(chat_html, unsafe_allow_html=True)
        st.markdown("""
        <script>
            const el = document.getElementById('chat-bottom');
            if (el) el.scrollIntoView({ behavior: 'smooth' });
        </script>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2.5rem 2rem">
            <div style="font-family:'DM Serif Display',serif;font-size:1.8rem;color:var(--line);margin-bottom:0.75rem">◈</div>
            <div style="font-size:0.82rem;color:var(--muted);font-weight:300;line-height:1.8">
                Ask anything about your meeting.<br>
                VAIA retrieves context before answering.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Input row
    q_col, btn_col = st.columns([6, 1], gap="small")
    with q_col:
        user_input = st.text_input(
            "chat_input",
            placeholder="What decisions were made?  Who owns the Q3 deliverable?",
            label_visibility="collapsed",
            key="chat_input_box",
            value=st.session_state.get("chat_input_val", "")
        )
    with btn_col:
        send_btn = st.button(
            "Sending…" if st.session_state.get("sending") else "Send",
            use_container_width=True,
            disabled=st.session_state.get("sending", False)
        )

    if send_btn and user_input.strip():
        question = user_input.strip()
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state["chat_input_val"] = ""
        try:
            with st.spinner("Retrieving context…"):
                answer = ask_question(r["rag_chain"], question)
        except Exception as e:
            answer = f"Sorry, something went wrong: {e}"
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear conversation", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

# ─── Empty State ────────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-glyph">◈</div>
        <div class="empty-heading">Nothing analysed yet</div>
        <div class="empty-sub">
            Paste a YouTube URL or a local file path into the sidebar,
            select your language, and press <strong style="color:var(--gold)">Run Analysis</strong>.
        </div>
        <div style="margin-top:2rem">
            <span class="pill pill-gold">YouTube URLs</span>
            <span class="pill pill-teal">MP4 · WAV · M4A</span>
            <span class="pill pill-rose">English · Hindi</span>
        </div>
    </div>
    """, unsafe_allow_html=True)