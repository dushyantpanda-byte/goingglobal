"""
Split extracted page JSON into ~500-word chunks with metadata.
Usage: python3 scripts/chunk.py <extracted_json> [output_json]

doc_type is inferred from the source filename. Override by passing --doc-type RBI etc.
"""
import json
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500        # words
CHUNK_OVERLAP = 50      # words

# Approx chars per word
CHARS_PER_WORD = 5

# Map source filename keywords → doc_type
DOC_TYPE_MAP = {
    "rbi": "RBI",
    "fema": "FEMA",
    "dgft": "DGFT",
    "ftp": "DGFT",
    "hop": "DGFT",
    "gst": "GST",
    "cbic": "GST",
    "exim": "EXIM",
    "ecgc": "ECGC",
    "nirvik": "ECGC",
    "customs": "Customs",
    "apeda": "DGFT",
}

# Source URL hints by doc_type (best-effort)
SOURCE_URL_MAP = {
    "RBI": "https://rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=10287",
    "FEMA": "https://rbi.org.in/Scripts/NotificationUser.aspx?Id=2612",
    "DGFT": "https://dgft.gov.in/CP/?opt=ftp",
    "GST": "https://cbic.gov.in",
    "EXIM": "https://eximbankindia.in",
    "ECGC": "https://ecgc.in",
    "Customs": "https://cbic.gov.in",
}


def infer_doc_type(source: str) -> str:
    source_lower = source.lower()
    for keyword, dtype in DOC_TYPE_MAP.items():
        if keyword in source_lower:
            return dtype
    return "Other"


def chunk_pages(pages: list[dict], doc_type_override: str | None = None) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE * CHARS_PER_WORD,
        chunk_overlap=CHUNK_OVERLAP * CHARS_PER_WORD,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page in pages:
        source = page["source"]
        doc_type = doc_type_override or infer_doc_type(source)
        source_url = SOURCE_URL_MAP.get(doc_type, "")

        texts = splitter.split_text(page["text"])
        for text in texts:
            chunks.append({
                "text": text,
                "doc_type": doc_type,
                "title": source.replace("_", " ").title(),
                "source_url": source_url,
                "page": page["page"],
                "source": source,
            })

    return chunks


def main():
    args = sys.argv[1:]
    doc_type_override = None

    if "--doc-type" in args:
        idx = args.index("--doc-type")
        doc_type_override = args[idx + 1]
        args = [a for i, a in enumerate(args) if i not in (idx, idx + 1)]

    if not args:
        print("Usage: python3 scripts/chunk.py <extracted_json> [output_json] [--doc-type TYPE]")
        sys.exit(1)

    input_path = Path(args[0])
    with open(input_path, encoding="utf-8") as f:
        pages = json.load(f)

    chunks = chunk_pages(pages, doc_type_override)

    output_path = Path(args[1]) if len(args) > 1 else input_path.parent / input_path.name.replace("_extracted", "_chunks")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Created {len(chunks)} chunks from {len(pages)} pages → {output_path}")


if __name__ == "__main__":
    main()
