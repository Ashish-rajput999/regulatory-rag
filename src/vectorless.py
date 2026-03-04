import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import call_groq
from src.config import GROQ_REASONING_MODEL, GROQ_FAST_MODEL, TREES_DIR, TOP_K_NODES

def load_tree(tree_path: Path) -> dict:
    with open(tree_path) as f:
        return json.load(f)

def get_available_trees() -> dict:
    """Returns {domain: [tree_paths]} for all available trees."""
    result = {}
    for domain_dir in TREES_DIR.iterdir():
        if domain_dir.is_dir():
            trees = list(domain_dir.glob("*.json"))
            if trees:
                result[domain_dir.name] = trees
    return result

def tree_to_toc_string(tree: dict) -> str:
    """Flatten tree into a readable TOC for the LLM."""
    lines = []
    for node in tree.get("nodes", []):
        node_id = node.get("node_id", "?")
        title = node.get("title", "Untitled")[:150]
        summary = node.get("summary", "")[:100]
        line = f"[{node_id}] {title}"
        if summary:
            line += f" — {summary}"
        lines.append(line)
    return "\n".join(lines)

def get_node_by_id(tree: dict, node_id: str) -> dict | None:
    """Find a node in the tree by its node_id."""
    for node in tree.get("nodes", []):
        if node.get("node_id") == node_id:
            return node
    return None

def pick_relevant_nodes(tree: dict, question: str, k: int = TOP_K_NODES) -> dict:
    """Step 1: LLM reasons over TOC and picks top-k relevant node_ids."""
    toc = tree_to_toc_string(tree)
    
    system = (
        "You are a legal research assistant specializing in Indian tax law. "
        "Given a table of contents and a question, select the most relevant sections. "
        "Respond ONLY with valid JSON, no markdown, no explanation."
    )
    
    prompt = f"""TABLE OF CONTENTS:
{toc}

QUESTION: {question}

Pick the {k} most relevant section IDs that would help answer this question.
Return ONLY this JSON:
{{"node_ids": ["0001", "0003"], "reasoning": "I chose these sections because..."}}"""

    raw = call_groq(
        prompt=prompt,
        model=GROQ_REASONING_MODEL,
        system=system,
        temperature=0.0,
        max_tokens=500
    )
    
    raw = raw.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    
    try:
        result = json.loads(raw.strip())
        return result
    except Exception as e:
        print(f"pick_relevant_nodes parse error: {e}, raw: {raw[:200]}")
        fallback_ids = [n["node_id"] for n in tree.get("nodes", [])[:k]]
        return {"node_ids": fallback_ids, "reasoning": "Fallback: used first sections"}

def answer_from_nodes(tree: dict, node_ids: list[str], question: str) -> dict:
    """Step 2: Generate grounded answer from selected node text."""
    nodes = [get_node_by_id(tree, nid) for nid in node_ids]
    nodes = [n for n in nodes if n]
    
    if not nodes:
        return {
            "answer": "Could not find relevant sections to answer this question.",
            "cited_nodes": [],
            "sections": []
        }
    
    context = "\n\n---\n\n".join(
        f"[Section {n['node_id']}] {n['title']}\n{n.get('text', n.get('summary', ''))}"
        for n in nodes
    )
    
    system = (
        "You are an Indian tax and regulatory expert. "
        "Answer strictly from the provided sections. "
        "Cite section numbers inline like [Section 0001]. "
        "If the answer is not in the provided sections, say so clearly."
    )
    
    prompt = f"""SECTIONS:
{context}

QUESTION: {question}

ANSWER (with inline citations):"""

    answer = call_groq(
        prompt=prompt,
        model=GROQ_REASONING_MODEL,
        system=system,
        temperature=0.1,
        max_tokens=800
    )
    
    return {
        "answer": answer,
        "cited_nodes": node_ids,
        "sections": [
            {"id": n["node_id"], "title": n["title"]}
            for n in nodes
        ]
    }

def vectorless_query(tree: dict, question: str) -> dict:
    """End-to-end: pick nodes by reasoning, then generate cited answer."""
    print(f"  Picking relevant nodes...")
    picked = pick_relevant_nodes(tree, question)
    print(f"  Selected nodes: {picked['node_ids']}")
    
    print(f"  Generating answer...")
    result = answer_from_nodes(tree, picked["node_ids"], question)
    result["reasoning_trace"] = picked.get("reasoning", "")
    
    return result

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    
    tree_path = TREES_DIR / "gst" / "circular_170_itc_reporting.json"
    if not tree_path.exists():
        print(f"Tree not found at {tree_path}")
        sys.exit(1)
    
    tree = load_tree(tree_path)
    print(f"Loaded tree: {tree['document']} ({tree['total_sections']} sections)\n")
    
    question = "What are the rules for reporting ineligible ITC in GSTR-3B?"
    print(f"Question: {question}\n")
    
    result = vectorless_query(tree, question)
    
    print(f"\n=== ANSWER ===")
    print(result["answer"])
    print(f"\n=== REASONING ===")
    print(result["reasoning_trace"])
    print(f"\n=== CITED SECTIONS ===")
    for s in result["sections"]:
        print(f"  [{s['id']}] {s['title']}")
