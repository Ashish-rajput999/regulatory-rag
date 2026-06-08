import sys
import os
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from src.vectorless import load_tree, vectorless_query, get_available_trees
from src.config import TREES_DIR

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GST Copilot",
    page_icon="⚖️",
    layout="wide"
)

# ── Google Fonts injection ────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# ── Load custom CSS ───────────────────────────────────────────────────────────
style_css_path = Path(__file__).parent / "style.css"
if style_css_path.exists():
    with open(style_css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ── Helper: lightweight markdown → HTML ──────────────────────────────────────
def md_to_html(text: str) -> str:
    """Convert basic markdown to inline HTML for card rendering."""
    # Bold and italic
    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'<strong><em>\1</em></strong>', text)
    text = re.sub(r'\*\*(.*?)\*\*',     r'<strong>\1</strong>',           text)
    text = re.sub(r'\*(.*?)\*',          r'<em>\1</em>',                   text)
    # Paragraphs from double newlines
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    html = ''.join(
        f'<p style="margin:0 0 14px 0;line-height:1.78;">{p.replace(chr(10), "<br>")}</p>'
        for p in paragraphs
    )
    return html or f'<p style="line-height:1.78;">{text}</p>'


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Logo / Wordmark ───────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:12px 0 24px 0;">
      <div style="display:flex;align-items:baseline;gap:7px;margin-bottom:4px;">
        <span style="font-family:'Playfair Display',serif;font-size:1.45rem;
                     font-weight:700;color:#FFFFFF;letter-spacing:-0.01em;">GST</span>
        <span style="font-family:'Playfair Display',serif;font-size:1.45rem;
                     font-weight:400;color:#B0B0C8;letter-spacing:-0.01em;">Copilot</span>
      </div>
      <div style="font-family:'DM Sans',sans-serif;font-size:0.68rem;color:#5A5A78;
                  letter-spacing:0.12em;text-transform:uppercase;">
        Vectorless RAG &nbsp;·&nbsp; GST Domain
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<hr style="border:none;border-top:1px solid rgba(232,232,240,0.1);margin:0 0 22px 0;">',
        unsafe_allow_html=True
    )

    # ── API Configuration ─────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-family:\'DM Sans\',sans-serif;font-size:0.68rem;font-weight:600;'
        'letter-spacing:0.13em;text-transform:uppercase;color:#5A5A78;margin-bottom:10px;">'
        'API Configuration</div>',
        unsafe_allow_html=True
    )

    try:
        api_key_from_env = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    except Exception:
        api_key_from_env = os.environ.get("GROQ_API_KEY", "")

    is_valid_env_key = bool(
        api_key_from_env and
        not api_key_from_env.startswith("your_groq_api_key_here")
    )

    if is_valid_env_key:
        # Also ensure os.environ is updated if it came from secrets
        os.environ["GROQ_API_KEY"] = api_key_from_env
        st.markdown(
            '<div class="api-badge success">&#9679;&nbsp; Groq key active (Secrets/Env)</div>',
            unsafe_allow_html=True
        )
        has_api_key = True
    else:
        api_key_from_state = st.session_state.get("groq_api_key", "")
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=api_key_from_state,
            placeholder="gsk_...",
            help="Used only during this session — never stored."
        )
        if api_key:
            api_key = api_key.strip()
            st.session_state["groq_api_key"] = api_key
            os.environ["GROQ_API_KEY"] = api_key
            st.markdown(
                '<div class="api-badge success">&#9679;&nbsp; Key applied for this session</div>',
                unsafe_allow_html=True
            )
            has_api_key = True
        else:
            st.markdown(
                '<div class="api-badge">&#9679;&nbsp; No API key configured</div>',
                unsafe_allow_html=True
            )
            has_api_key = False

    st.markdown(
        '<hr style="border:none;border-top:1px solid rgba(232,232,240,0.1);margin:22px 0;">',
        unsafe_allow_html=True
    )

    # ── How it works — timeline ───────────────────────────────────────────────
    with st.expander("How it works"):
        st.markdown("""
        <div class="timeline-step">
          <div class="timeline-num">1</div>
          <div class="timeline-text">
            <strong>PDF → Tree</strong>
            Document parsed into a section hierarchy (one-time build)
          </div>
        </div>
        <div class="timeline-step">
          <div class="timeline-num">2</div>
          <div class="timeline-text">
            <strong>Question → TOC</strong>
            LLM reads the table of contents to understand structure
          </div>
        </div>
        <div class="timeline-step">
          <div class="timeline-num">3</div>
          <div class="timeline-text">
            <strong>Reasoning</strong>
            Groq picks the most relevant sections by title
          </div>
        </div>
        <div class="timeline-step">
          <div class="timeline-num">4</div>
          <div class="timeline-text">
            <strong>Answer</strong>
            Grounded response with cited section references
          </div>
        </div>
        <div class="no-embeddings-note">
          No embeddings &nbsp;·&nbsp; No vector DB &nbsp;·&nbsp; Pure reasoning
        </div>
        """, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:40px;padding-top:18px;
                border-top:1px solid rgba(232,232,240,0.1);">
      <div style="font-family:'DM Sans',sans-serif;font-size:0.78rem;
                  color:#5A5A78;margin-bottom:4px;">Built by ashish singh naruka</div>
      <a href="https://github.com/Ashish-rajput999" target="_blank"
         style="font-family:'DM Sans',sans-serif;font-size:0.78rem;
                color:#C84B00;text-decoration:none;letter-spacing:0.01em;">
        GitHub &rarr;
      </a>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["Ask", "Explore Tree", "Benchmark"])


# ── TAB 1: ASK ────────────────────────────────────────────────────────────────
with tab1:

    st.markdown("""
    <div style="padding:20px 0 28px 0;">
      <h1 style="font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;
                 color:#1C1917;margin:0 0 8px 0;letter-spacing:-0.01em;">
        Ask a Regulatory Question
      </h1>
      <p style="font-family:'DM Sans',sans-serif;font-size:0.9rem;color:#78716C;margin:0;
                line-height:1.6;">
        Search across Indian GST circulars using tree-based reasoning — no vector embeddings.
      </p>
    </div>
    """, unsafe_allow_html=True)

    available_trees = get_available_trees()

    if not available_trees:
        st.warning("No document trees found. Run `python scripts/build_trees.py` first.")
    else:
        # Domain + Document selects — one row
        col1, col2 = st.columns(2)
        with col1:
            domain = st.selectbox("Domain", options=list(available_trees.keys()))
        with col2:
            tree_files = available_trees.get(domain, [])
            tree_names = {f.stem: f for f in tree_files}
            selected_name = st.selectbox("Document", options=list(tree_names.keys()))

        # Example question chips
        st.markdown("""
        <div style="font-family:'DM Sans',sans-serif;font-size:0.72rem;font-weight:600;
                    letter-spacing:0.11em;text-transform:uppercase;color:#78716C;
                    margin:24px 0 10px 0;">
          Try an example
        </div>
        """, unsafe_allow_html=True)

        examples = [
            "What are the rules for reporting ineligible ITC in GSTR-3B?",
            "How should ITC reversals be reported in Table 4 of GSTR-3B?",
            "What is the correct procedure for inter-state supply reporting?"
        ]
        ecol1, ecol2, ecol3 = st.columns(3)
        if ecol1.button(f"{examples[0][:44]}…", key="ex1"):
            st.session_state.question = examples[0]
            st.rerun()
        if ecol2.button(f"{examples[1][:44]}…", key="ex2"):
            st.session_state.question = examples[1]
            st.rerun()
        if ecol3.button(f"{examples[2][:44]}…", key="ex3"):
            st.session_state.question = examples[2]
            st.rerun()

        # Question textarea
        st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
        question = st.text_area(
            "Your question",
            value=st.session_state.get("question", ""),
            height=96,
            placeholder="e.g. What is the time limit for availing ITC under Section 16(4)?",
            label_visibility="collapsed"
        )
        st.markdown(
            '<div style="margin-top:-6px;margin-bottom:18px;font-family:\'DM Sans\',sans-serif;'
            'font-size:0.78rem;color:#A8A29E;">Type your question or select an example above</div>',
            unsafe_allow_html=True
        )

        # Missing API key nudge
        if not has_api_key:
            st.markdown("""
            <div style="padding:11px 16px;background:#FFF8F4;border:1px solid #FDDCC8;
                        border-radius:6px;font-family:'DM Sans',sans-serif;font-size:0.85rem;
                        color:#78716C;margin-bottom:18px;">
              Configure your Groq API key in the sidebar to enable querying.
            </div>
            """, unsafe_allow_html=True)

        # Ask button
        ask_clicked = st.button(
            "Ask",
            type="primary",
            disabled=not question or not has_api_key
        )

        if ask_clicked:
            tree_path = tree_names[selected_name]
            tree = load_tree(tree_path)

            with st.spinner("Reasoning over document tree…"):
                try:
                    result = vectorless_query(tree, question)

                    # ── Answer card ───────────────────────────────────────────
                    answer_html = md_to_html(result["answer"])
                    st.markdown(f"""
                    <div class="answer-card">
                        <div class="answer-label">Answer</div>
                        <div class="answer-text">{answer_html}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Reasoning Trace + Cited Sections (side-by-side) ───────
                    col_a, col_b = st.columns(2)

                    with col_a:
                        reasoning_raw = result.get("reasoning_trace", "")
                        reasoning_html = (
                            md_to_html(reasoning_raw)
                            if reasoning_raw
                            else '<p style="color:#A8A29E;font-size:0.87rem;">No trace available.</p>'
                        )
                        st.markdown(f"""
                        <div class="trace-card">
                            <div class="card-label">Reasoning Trace</div>
                            <div style="font-family:'DM Sans',sans-serif;font-size:0.87rem;
                                        color:#44403C;line-height:1.65;">
                                {reasoning_html}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_b:
                        sections = result.get("sections", [])
                        if sections:
                            citations_inner = "".join(
                                f'<div class="citation-box">'
                                f'<span class="citation-id">{s["id"]}</span>'
                                f'<span>{s["title"]}</span>'
                                f'</div>'
                                for s in sections
                            )
                        else:
                            citations_inner = (
                                '<p style="color:#A8A29E;font-size:0.87rem;">'
                                'No sections cited.</p>'
                            )
                        st.markdown(f"""
                        <div class="cite-card">
                            <div class="card-label">Cited Sections</div>
                            {citations_inner}
                        </div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    st.error("Failed to query the Groq model. Check your API key and try again.")
                    st.exception(e)


# ── TAB 2: EXPLORE TREE ───────────────────────────────────────────────────────
with tab2:

    st.markdown("""
    <div style="padding:20px 0 28px 0;">
      <h1 style="font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;
                 color:#1C1917;margin:0 0 8px 0;letter-spacing:-0.01em;">
        Explore Document Tree
      </h1>
      <p style="font-family:'DM Sans',sans-serif;font-size:0.9rem;color:#78716C;margin:0;
                line-height:1.6;">
        Browse the parsed section hierarchy of any loaded regulatory document.
      </p>
    </div>
    """, unsafe_allow_html=True)

    available_trees2 = get_available_trees()

    if not available_trees2:
        st.warning("No trees available yet. Run `python scripts/build_trees.py` first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            # trailing space avoids key collision with tab1 selectbox
            domain2 = st.selectbox("Domain ", options=list(available_trees2.keys()))
        with col2:
            tree_files2 = available_trees2.get(domain2, [])
            tree_names2 = {f.stem: f for f in tree_files2}
            selected_name2 = st.selectbox("Document ", options=list(tree_names2.keys()))

        tree2 = load_tree(tree_names2[selected_name2])
        n_sections = tree2.get("total_sections", len(tree2.get("nodes", [])))

        st.markdown(
            f'<div class="section-badge">&#9632;&nbsp; {n_sections} sections</div>',
            unsafe_allow_html=True
        )

        for node in tree2.get("nodes", []):
            node_id   = node.get("node_id", "")
            title     = node.get("title", "Untitled")[:80]
            with st.expander(f"[{node_id}]  {title}"):
                if node.get("summary"):
                    st.caption(node["summary"])
                if node.get("text"):
                    st.markdown(
                        f'<div style="font-family:\'DM Sans\',sans-serif;font-size:0.9rem;'
                        f'color:#1C1917;line-height:1.7;">{node["text"]}</div>',
                        unsafe_allow_html=True
                    )


# ── TAB 3: BENCHMARK ──────────────────────────────────────────────────────────
with tab3:

    st.markdown("""
    <div style="padding:20px 0 28px 0;">
      <h1 style="font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;
                 color:#1C1917;margin:0 0 8px 0;letter-spacing:-0.01em;">
        Benchmark
      </h1>
      <p style="font-family:'DM Sans',sans-serif;font-size:0.9rem;color:#78716C;margin:0;
                line-height:1.6;">
        Compare vectorless retrieval against traditional FAISS-based embeddings.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="coming-soon-card">
        <div class="coming-soon-icon">&#9638;</div>
        <div class="coming-soon-title">Benchmark Suite — Coming Soon</div>
        <div class="coming-soon-desc">
            We&rsquo;re building a structured evaluation to compare tree-based reasoning
            against FAISS embeddings on precision, latency, and citation accuracy
            across Indian GST regulatory documents.
        </div>
        <div class="feature-pills">
            <span class="feature-pill">Precision @ K</span>
            <span class="feature-pill">Latency comparison</span>
            <span class="feature-pill">Citation accuracy</span>
            <span class="feature-pill">Token efficiency</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
