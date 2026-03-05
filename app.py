import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import json
from dotenv import load_dotenv
load_dotenv()

from src.vectorless import load_tree, vectorless_query, get_available_trees
from src.config import TREES_DIR

# Page config
st.set_page_config(
    page_title="GST Copilot",
    page_icon="⚖️",
    layout="wide"
)

# Load Custom Styling
style_css_path = Path(__file__).parent / "style.css"
if style_css_path.exists():
    with open(style_css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚖️ GST Copilot")
    st.caption("Vectorless RAG for Indian Regulatory Docs")
    st.divider()
    
    # API Key Configuration
    st.subheader("🔑 API Configuration")
    api_key_from_env = os.environ.get("GROQ_API_KEY", "")
    is_valid_env_key = api_key_from_env and not api_key_from_env.startswith("your_groq_api_key_here")
    
    if is_valid_env_key:
        st.success("Groq API Key detected in `.env`")
        has_api_key = True
    else:
        api_key_from_state = st.session_state.get("groq_api_key", "")
        api_key = st.text_input(
            "Enter Groq API Key",
            type="password",
            value=api_key_from_state,
            help="Your API Key is only used during this session and is not stored."
        )
        if api_key:
            api_key = api_key.strip()
            st.session_state["groq_api_key"] = api_key
            os.environ["GROQ_API_KEY"] = api_key
            st.success("API Key applied!")
            has_api_key = True
        else:
            st.warning("⚠️ Groq API key is missing. Please configure it in `.env` or enter it above to query.")
            has_api_key = False

    st.divider()
    
    with st.expander("🧠 How it works"):
        st.markdown("""
        1. **PDF → Tree**: Document parsed into section hierarchy (one-time)
        2. **Question → TOC**: LLM reads table of contents
        3. **Reasoning**: Groq picks most relevant sections
        4. **Answer**: Grounded response with section citations
        
        **No embeddings. No vector DB. Pure reasoning.**
        """)
    st.divider()
    st.caption("Built by Pushkar Sharma")
    st.caption("[GitHub](https://github.com/pushkar462)")

# Main tabs
tab1, tab2, tab3 = st.tabs(["💬 Ask", "🌳 Explore Tree", "📊 Benchmark"])

# ── TAB 1: ASK ──────────────────────────────────────────────────────────────
with tab1:
    st.header("Ask a Regulatory Question")

    available_trees = get_available_trees()

    if not available_trees:
        st.warning("No document trees found. Run `python src/tree_builder.py` first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            domain = st.selectbox("Domain", options=list(available_trees.keys()))
        with col2:
            tree_files = available_trees.get(domain, [])
            tree_names = {f.stem: f for f in tree_files}
            selected_name = st.selectbox("Document", options=list(tree_names.keys()))

        st.markdown("**Try an example:**")
        examples = [
            "What are the rules for reporting ineligible ITC in GSTR-3B?",
            "How should ITC reversals be reported in Table 4 of GSTR-3B?",
            "What is the correct procedure for inter-state supply reporting?"
        ]
        ecol1, ecol2, ecol3 = st.columns(3)
        if ecol1.button(f"📌 {examples[0][:40]}..."):
            st.session_state.question = examples[0]
            st.rerun()
        if ecol2.button(f"📌 {examples[1][:40]}..."):
            st.session_state.question = examples[1]
            st.rerun()
        if ecol3.button(f"📌 {examples[2][:40]}..."):
            st.session_state.question = examples[2]
            st.rerun()

        question = st.text_area(
            "Your question",
            value=st.session_state.get("question", ""),
            height=100,
            placeholder="e.g. What is the time limit for availing ITC under Section 16(4)?"
        )

        if not has_api_key:
            st.info("💡 To ask questions, configure your Groq API key in the sidebar.")

        if st.button("🔍 Ask", type="primary", disabled=not question or not has_api_key):
            tree_path = tree_names[selected_name]
            tree = load_tree(tree_path)

            with st.spinner("Reasoning over document tree..."):
                try:
                    result = vectorless_query(tree, question)
                    st.success("Answer ready!")
                    st.markdown("### 📋 Answer")
                    st.markdown(result["answer"])

                    col_a, col_b = st.columns(2)
                    with col_a:
                        with st.expander("🧠 Reasoning Trace", expanded=True):
                            st.markdown(result.get("reasoning_trace", "N/A"))
                    with col_b:
                        with st.expander("📎 Cited Sections", expanded=True):
                            for s in result.get("sections", []):
                                st.markdown(f'<div class="citation-box"><strong>[{s["id"]}]</strong> {s["title"]}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error("Failed to query Groq model.")
                    st.exception(e)

# ── TAB 2: EXPLORE TREE ─────────────────────────────────────────────────────
with tab2:
    st.header("Explore Document Tree")

    available_trees2 = get_available_trees()
    if not available_trees2:
        st.warning("No trees available yet.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            domain2 = st.selectbox("Domain ", options=list(available_trees2.keys()))
        with col2:
            tree_files2 = available_trees2.get(domain2, [])
            tree_names2 = {f.stem: f for f in tree_files2}
            selected_name2 = st.selectbox("Document ", options=list(tree_names2.keys()))

        tree2 = load_tree(tree_names2[selected_name2])
        st.caption(f"📄 {tree2['total_sections']} sections found")

        for node in tree2.get("nodes", []):
            with st.expander(f"[{node['node_id']}] {node['title'][:80]}"):
                if node.get("summary"):
                    st.caption(node["summary"])
                if node.get("text"):
                    st.write(node["text"])

# ── TAB 3: BENCHMARK ────────────────────────────────────────────────────────
with tab3:
    st.header("Benchmark")
    st.markdown("Use this tab to compare response times or validate tree-based answers.")
    st.info("Benchmark support is coming soon. Run the app with existing trees and ask questions using the Ask tab.")
