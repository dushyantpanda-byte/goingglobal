"""
Extract text from a PDF file, page by page, saving to JSON.
Usage: python3 scripts/extract.py <path_to_pdf> [output_json]
"""
import json
import sys
from pathlib import Path

import pdfplumber

MIN_CHARS = 50


def extract(pdf_path: str, output_path: str | None = None) -> list[dict]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"ERROR: file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    source = pdf_path.stem
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"Extracting {total} pages from {pdf_path.name} ...")
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            text = text.strip()
            if len(text) < MIN_CHARS:
                continue
            pages.append({"page": i, "text": text, "source": source})

    print(f"  Kept {len(pages)} pages (skipped {total - len(pages)} with < {MIN_CHARS} chars)")

    if output_path is None:
        output_path = Path("data") / f"{source}_extracted.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    print(f"  Saved to {output_path}")
    return pages


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/extract.py <pdf_path> [output_json]")
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) > 2 else None
    extract(sys.argv[1], out)
