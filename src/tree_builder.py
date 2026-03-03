import json
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pypdf import PdfReader
from src.llm import call_groq
from src.config import GROQ_FAST_MODEL, GROQ_REASONING_MODEL

def extract_pages(pdf_path: str) -> list[dict]:
    """Extract text from each page of a PDF."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page_num": i + 1, "text": text.strip()})
    return pages

def chunk_pages(pages: list[dict], chunk_size: int = 3) -> list[dict]:
    """Group pages into chunks for processing."""
    chunks = []
    for i in range(0, len(pages), chunk_size):
        group = pages[i:i + chunk_size]
        combined_text = "\n\n".join(
            f"[Page {p['page_num']}]\n{p['text'][:800]}" for p in group
        )
        chunks.append({
            "start_page": group[0]["page_num"],
            "end_page": group[-1]["page_num"],
            "text": combined_text
        })
    return chunks

def extract_sections_from_chunk(chunk: dict, doc_type: str = "GST regulatory document") -> list[dict]:
    """Ask Groq to identify sections in a chunk of pages."""
    prompt = f"""You are analyzing a {doc_type}.
Extract all section headers and their content from these pages.

TEXT:
{chunk['text'][:2500]}

Return ONLY a JSON array like this:
[
  {{
    "node_id": "001",
    "title": "Section 16 - Eligibility and conditions for taking input tax credit",
    "page_start": {chunk['start_page']},
    "page_end": {chunk['end_page']},
    "summary": "One sentence summary of this section",
    "text": "First 300 chars of section content"
  }}
]

Rules:
- Only include actual section headers (not every paragraph)
- If no clear sections found, return empty array []
- Return ONLY the JSON array, no markdown, no explanation"""

    try:
        response = call_groq(
            prompt=prompt,
            model=GROQ_FAST_MODEL,
            temperature=0.0,
            max_tokens=1000
        )
        # Clean response
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()
        sections = json.loads(response)
        return sections if isinstance(sections, list) else []
    except Exception as e:
        print(f"  Warning: chunk {chunk['start_page']}-{chunk['end_page']} parse failed: {e}")
        return []

def build_tree_from_sections(sections: list[dict], pdf_name: str) -> dict:
    """Build a hierarchical tree from flat sections list."""
    # Add unique node IDs
    for i, section in enumerate(sections):
        section["node_id"] = str(i + 1).zfill(4)
        section["nodes"] = []  # children placeholder

    return {
        "document": pdf_name,
        "total_sections": len(sections),
        "nodes": sections
    }

def build_tree_for_pdf(pdf_path: Path, output_dir: Path, model: str = None) -> Path:
    """Main entry point: build tree for a PDF and save to output_dir."""
    print(f"\nBuilding tree for: {pdf_path.name}")
    
    # Extract pages
    pages = extract_pages(str(pdf_path))
    print(f"  Extracted {len(pages)} pages")
    
    # Chunk pages
    chunks = chunk_pages(pages, chunk_size=3)
    print(f"  Processing {len(chunks)} chunks...")
    
    # Extract sections from each chunk
    all_sections = []
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}/{len(chunks)} (pages {chunk['start_page']}-{chunk['end_page']})...")
        sections = extract_sections_from_chunk(chunk)
        all_sections.extend(sections)
        print(f"    Found {len(sections)} sections")
        time.sleep(2)  # respect rate limits
    
    print(f"  Total sections found: {len(all_sections)}")
    
    # Build tree
    tree = build_tree_from_sections(all_sections, pdf_path.stem)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{pdf_path.stem}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ Tree saved to {output_file}")
    return output_file

if __name__ == "__main__":
    # Quick test
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    
    pdf_path = Path("data/raw_pdfs/gst/circular_170_itc_reporting.pdf")
    output_dir = Path("data/trees/gst")
    result = build_tree_for_pdf(pdf_path, output_dir)
    print(f"\nDone! Tree at: {result}")
    
    # Preview
    with open(result) as f:
        tree = json.load(f)
    print(f"Document: {tree['document']}")
    print(f"Sections: {tree['total_sections']}")
    for node in tree['nodes'][:3]:
        print(f"  [{node['node_id']}] {node['title']}")
