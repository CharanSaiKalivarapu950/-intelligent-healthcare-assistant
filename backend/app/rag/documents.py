from pathlib import Path
from pypdf import PdfReader

def extract_pdf(path: str | Path):
    path = Path(path)
    reader = PdfReader(str(path))
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({
                "text": text,
                "source": path.name,
                "page": page_num,
            })
    return pages

def chunk_page(text: str, chunk_size=700, overlap=120):
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
    chunks, start = [], 0
    text = " ".join(text.split())
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks

def load_pdf_chunks(directory="medical_documents/raw"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for pdf in sorted(directory.glob("*.pdf")):
        for page in extract_pdf(pdf):
            for chunk_idx, chunk in enumerate(chunk_page(page["text"])):
                records.append({
                    "text": chunk,
                    "source": page["source"],
                    "page": page["page"],
                    "chunk": chunk_idx,
                })
    return records
