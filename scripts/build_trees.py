import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from dotenv import load_dotenv
load_dotenv()

from src.tree_builder import build_tree_for_pdf
from src.config import PDFS_DIR, TREES_DIR, GROQ_REASONING_MODEL

def main():
    pdf_files = list(PDFS_DIR.rglob("*.pdf"))
    if not pdf_files:
        print("⚠️  No PDFs found in data/raw_pdfs/")
        print("Please add PDFs to data/raw_pdfs/gst/, data/raw_pdfs/rbi/, data/raw_pdfs/incometax/")
        return

    print(f"Found {len(pdf_files)} PDFs to process...")
    success, failed = [], []

    for pdf_path in pdf_files:
        domain = pdf_path.parent.name
        output_dir = TREES_DIR / domain
        try:
            dest = build_tree_for_pdf(pdf_path, output_dir, GROQ_REASONING_MODEL)
            success.append(dest)
        except Exception as e:
            print(f"❌ Failed: {pdf_path.name} — {e}")
            failed.append(pdf_path.name)

    print(f"\n✅ Built {len(success)} trees")
    if failed:
        print(f"❌ Failed: {failed}")

if __name__ == "__main__":
    main()
