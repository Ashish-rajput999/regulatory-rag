import sys
import os
import re
import time
import json as _json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from src.vectorless import load_tree, vectorless_query
from src.config import TREES_DIR, PDFS_DIR

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


# ── Helper: always-fresh tree discovery (never cached) ───────────────────────
def get_available_trees() -> dict:
    """Scans TREES_DIR fresh every call — picks up newly uploaded docs."""
    result = {}
    if not TREES_DIR.exists():
        return result
    for domain_dir in sorted(TREES_DIR.iterdir()):
        if domain_dir.is_dir():
            trees = sorted(domain_dir.glob("*.json"))
            # Only include trees that have at least 1 section
            valid = []
            for t in trees:
                try:
                    data = _json.loads(t.read_text(encoding="utf-8"))
                    if data.get("total_sections", 0) > 0:
                        valid.append(t)
                except Exception:
                    pass
            if valid:
                result[domain_dir.name] = valid
    return result


# ── Helper: lightweight markdown → HTML ──────────────────────────────────────
def md_to_html(text: str) -> str:
    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'<strong><em>\1</em></strong>', text)
    text = re.sub(r'\*\*(.*?)\*\*',     r'<strong>\1</strong>',           text)
    text = re.sub(r'\*(.*?)\*',          r'<em>\1</em>',                   text)
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

    st.markdown("""
    <div style="margin-top:40px;padding-top:18px;
                border-top:1px solid rgba(232,232,240,0.1);">
      <div style="font-family:'DM Sans',sans-serif;font-size:0.78rem;
                  color:#5A5A78;margin-bottom:4px;">Built by Ashish</div>
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
tab1, tab2, tab3, tab4 = st.tabs(["Ask", "Explore Tree", "Benchmark", "Upload Document"])


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
        Search across regulatory documents using tree-based reasoning — no vector embeddings.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Always read fresh from disk so newly uploaded docs appear immediately
    available_trees = get_available_trees()

    if not available_trees:
        st.warning("No document trees found. Upload a document in the **Upload Document** tab to get started.")
    else:
        all_domains = list(available_trees.keys())

        # Pre-select newly uploaded domain/doc by injecting into widget keys directly,
        # then immediately clear so the dropdowns stay fully interactive afterwards.
        if "uploaded_domain" in st.session_state:
            wanted = st.session_state.pop("uploaded_domain")
            if wanted in all_domains:
                st.session_state["ask_domain"] = wanted

        col1, col2 = st.columns(2)
        with col1:
            domain = st.selectbox("Domain", options=all_domains, key="ask_domain")

        with col2:
            tree_files = available_trees.get(domain, [])
            tree_names = {f.stem: f for f in tree_files}
            all_docs   = list(tree_names.keys())

            if "uploaded_doc" in st.session_state:
                wanted_doc = st.session_state.pop("uploaded_doc")
                if wanted_doc in all_docs:
                    st.session_state["ask_doc"] = wanted_doc

            selected_name = st.selectbox("Document", options=all_docs, key="ask_doc")

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

        if not has_api_key:
            st.markdown("""
            <div style="padding:11px 16px;background:#FFF8F4;border:1px solid #FDDCC8;
                        border-radius:6px;font-family:'DM Sans',sans-serif;font-size:0.85rem;
                        color:#78716C;margin-bottom:18px;">
              Configure your Groq API key in the sidebar to enable querying.
            </div>
            """, unsafe_allow_html=True)

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

                    answer_html = md_to_html(result["answer"])
                    st.markdown(f"""
                    <div class="answer-card">
                        <div class="answer-label">Answer</div>
                        <div class="answer-text">{answer_html}</div>
                    </div>
                    """, unsafe_allow_html=True)

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
        st.warning("No trees available yet. Upload a document in the **Upload Document** tab.")
    else:
        col1, col2 = st.columns(2)
        with col1:
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
            node_id = node.get("node_id", "")
            title   = node.get("title", "Untitled")[:80]
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


# ── TAB 4: UPLOAD DOCUMENT ────────────────────────────────────────────────────
with tab4:

    st.markdown("""
    <div style="padding:20px 0 28px 0;">
      <h1 style="font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;
                 color:#1C1917;margin:0 0 8px 0;letter-spacing:-0.01em;">
        Upload a Document
      </h1>
      <p style="font-family:'DM Sans',sans-serif;font-size:0.9rem;color:#78716C;margin:0;
                line-height:1.6;">
        Upload any regulatory PDF. The system parses it into a section tree so you can
        query it immediately from the <strong>Ask</strong> tab — no code required.
      </p>
    </div>
    """, unsafe_allow_html=True)

    if not has_api_key:
        st.markdown("""
        <div style="padding:14px 18px;background:#FFF8F4;border:1px solid #FDDCC8;
                    border-radius:8px;font-family:'DM Sans',sans-serif;font-size:0.9rem;
                    color:#78716C;margin-bottom:24px;">
          ⚠️&nbsp; <strong>Groq API key required.</strong> Configure it in the sidebar first —
          the AI needs it to parse your document.
        </div>
        """, unsafe_allow_html=True)

    # ── Step 1 ─────────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;font-weight:600;'
        'letter-spacing:0.11em;text-transform:uppercase;color:#78716C;margin-bottom:10px;">'
        'Step 1 — Choose a domain</div>',
        unsafe_allow_html=True
    )

    PRESET_DOMAINS = ["gst", "incometax", "rbi", "sebi", "custom"]
    domain_choice = st.selectbox(
        "Domain",
        options=PRESET_DOMAINS,
        key="upload_domain_select",
        help="The regulatory category this document belongs to."
    )

    if domain_choice == "custom":
        custom_domain_input = st.text_input(
            "Custom domain name",
            placeholder="e.g. customs, fema, companies_act",
            help="Lowercase letters and underscores only.",
            key="custom_domain_input"
        ).strip().lower().replace(" ", "_")
        final_domain = custom_domain_input if custom_domain_input else None
    else:
        final_domain = domain_choice

    # ── Step 2 ─────────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;font-weight:600;'
        'letter-spacing:0.11em;text-transform:uppercase;color:#78716C;'
        'margin:24px 0 10px 0;">Step 2 — Upload your PDF</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Drop a PDF here",
        type=["pdf"],
        label_visibility="collapsed",
        key="pdf_uploader"
    )

    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.markdown(
            f'<div style="font-family:\'DM Sans\',sans-serif;font-size:0.82rem;color:#78716C;'
            f'margin-top:8px;">📄&nbsp; <strong>{uploaded_file.name}</strong>'
            f'&nbsp;·&nbsp; {file_size_mb:.1f} MB</div>',
            unsafe_allow_html=True
        )

    # ── Step 3 ─────────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-family:\'DM Sans\',sans-serif;font-size:0.72rem;font-weight:600;'
        'letter-spacing:0.11em;text-transform:uppercase;color:#78716C;'
        'margin:24px 0 10px 0;">Step 3 — Process document</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div style="padding:12px 16px;background:rgba(200,75,0,0.06);border-left:3px solid #C84B00;
                border-radius:0 6px 6px 0;font-family:'DM Sans',sans-serif;font-size:0.85rem;
                color:#44403C;margin-bottom:20px;line-height:1.65;">
      Clicking <strong>Process Document</strong> will:<br>
      &nbsp;&nbsp;① Save the PDF to the server's document library<br>
      &nbsp;&nbsp;② Send it to the AI in chunks to extract section headers<br>
      &nbsp;&nbsp;③ Build a JSON tree — takes <strong>1–3 minutes</strong> depending on PDF length<br>
      &nbsp;&nbsp;④ Make it instantly available in the <strong>Ask</strong> tab
    </div>
    """, unsafe_allow_html=True)

    can_process = bool(uploaded_file and final_domain and has_api_key)
    process_clicked = st.button("Process Document", type="primary", disabled=not can_process)

    if not can_process:
        if not uploaded_file:
            st.caption("Upload a PDF above to enable processing.")
        elif not final_domain:
            st.caption("Enter a custom domain name above.")
        elif not has_api_key:
            st.caption("Add your Groq API key in the sidebar.")

    # ── Processing ─────────────────────────────────────────────────────────────
    if process_clicked and can_process:

        safe_stem    = Path(uploaded_file.name).stem.replace(" ", "_").lower()
        pdf_dest_dir = PDFS_DIR / final_domain
        pdf_dest_dir.mkdir(parents=True, exist_ok=True)
        pdf_dest     = pdf_dest_dir / f"{safe_stem}.pdf"
        tree_dest_dir = TREES_DIR / final_domain
        tree_dest    = tree_dest_dir / f"{safe_stem}.json"

        # Duplicate guard
        if tree_dest.exists():
            st.warning(
                f"**{safe_stem}** already exists in domain **{final_domain}**. "
                "Rename your PDF or delete the existing tree to re-process."
            )
            st.stop()

        # Save PDF bytes to disk
        with open(pdf_dest, "wb") as f:
            f.write(uploaded_file.getbuffer())

        progress_bar = st.progress(0, text="Starting…")
        status_box   = st.empty()

        try:
            from src.tree_builder import (
                extract_pages,
                chunk_pages,
                extract_sections_from_chunk,
                build_tree_from_sections,
            )

            # A — read pages
            progress_bar.progress(5, text="Reading PDF pages…")
            status_box.info("📖 Reading PDF pages…")
            pages = extract_pages(str(pdf_dest))

            if not pages:
                raise ValueError("Could not extract any text from this PDF. It may be scanned/image-only.")

            # B — chunk
            progress_bar.progress(15, text=f"Found {len(pages)} pages — chunking…")
            chunks = chunk_pages(pages, chunk_size=3)
            n_chunks = len(chunks)

            # C — AI section extraction per chunk
            all_sections = []
            for i, chunk in enumerate(chunks):
                pct   = 15 + int(70 * (i / n_chunks))
                label = f"AI parsing chunk {i+1}/{n_chunks}  (pages {chunk['start_page']}–{chunk['end_page']})…"
                progress_bar.progress(pct, text=label)
                status_box.info(f"⚙️ {label}")
                sections = extract_sections_from_chunk(chunk)
                all_sections.extend(sections)
                time.sleep(2)

            # D — build + save tree
            progress_bar.progress(90, text="Building section tree…")
            status_box.info("🌲 Building section tree…")
            tree = build_tree_from_sections(all_sections, safe_stem)

            tree_dest_dir.mkdir(parents=True, exist_ok=True)
            with open(tree_dest, "w", encoding="utf-8") as f:
                _json.dump(tree, f, indent=2, ensure_ascii=False)

            progress_bar.progress(100, text="Done!")
            status_box.empty()

            # Zero-sections warning (e.g. non-regulatory PDF)
            if tree["total_sections"] == 0:
                # Delete the empty tree so it doesn't pollute the Ask dropdown
                tree_dest.unlink(missing_ok=True)
                pdf_dest.unlink(missing_ok=True)
                st.warning(
                    "⚠️ The AI could not find any section headers in this document. "
                    "This usually happens with:\n"
                    "- Scanned / image PDFs (no selectable text)\n"
                    "- Documents without clear section headings\n\n"
                    "Try a proper regulatory circular or act with numbered sections."
                )
            else:
                # ── Store hints so Ask tab auto-selects this doc ───────────────
                st.session_state["uploaded_domain"] = final_domain
                st.session_state["uploaded_doc"]    = safe_stem

                st.markdown(f"""
                <div style="padding:18px 22px;background:#F0FDF4;border:1px solid #BBF7D0;
                            border-radius:8px;font-family:'DM Sans',sans-serif;
                            font-size:0.92rem;color:#14532D;margin-top:16px;line-height:1.7;">
                  ✅&nbsp; <strong>{safe_stem}</strong> processed successfully!<br>
                  <span style="color:#166534;">
                    {tree['total_sections']} sections extracted
                    &nbsp;·&nbsp; domain: <strong>{final_domain}</strong>
                  </span><br><br>
                  👉 Click the <strong>Ask</strong> tab — your document is already selected.
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            progress_bar.empty()
            status_box.empty()
            # Clean up partial files
            if pdf_dest.exists():
                pdf_dest.unlink()
            if tree_dest.exists():
                tree_dest.unlink()
            st.error("Document processing failed. Files have been cleaned up.")
            st.exception(e)
