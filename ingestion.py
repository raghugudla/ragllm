import os
import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

PERSIST_DIR = "./data/chroma_db"
SOURCES_DIR = "./data/sources"
client = chromadb.PersistentClient(path=PERSIST_DIR)

collection = client.get_or_create_collection("pdf_collection")
model = SentenceTransformer('all-MiniLM-L6-v2')

def ingest_units_to_chroma(units, doc_id_prefix):
    for unit in units:
        unit_id = f"{doc_id_prefix}_{unit['id']}"
        existing = collection.get(ids=[unit_id])
        if existing.get("ids"):
            print(f"Unit {unit_id} already ingested. Skipping.")
            continue
        
        embedding = model.encode([unit["content"]])
        collection.upsert(
            ids=[unit_id],
            documents=[unit["content"]],
            embeddings=embedding.tolist(),
            metadatas=[{
                "docid": doc_id_prefix,
                "unit_id": unit_id,
                "modality": unit.get("modality", "unknown")
            }]
        )
        print(f"Ingested unit {unit_id}")

def extract_multimodal_units_from_pdf(file_path):
    # Simplified example: returns list of dicts with extracted text per page
    units = []
    reader = PdfReader(file_path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        units.append({
            "id": f"page{i}_unit0",  # removed file_path from id, doc_id_prefix handles uniqueness
            "content": text.strip(),
            "modality": "text",
            "entity_summary": None
        })
    return units

def get_pdf_title_version(pdf_path):
    reader = PdfReader(pdf_path)
    meta = reader.metadata  # new API

    title = meta.title if meta and meta.title else "unknown_title"
    mod_time = os.path.getmtime(pdf_path)
    version = str(int(mod_time))
    title = title.replace(" ", "_")
    return title, version

def ingest_pdf_to_chroma(pdf_path):
    title, version = get_pdf_title_version(pdf_path)
    doc_id_prefix = f"{title}_v{version}"
    units = extract_multimodal_units_from_pdf(pdf_path)
    ingest_units_to_chroma(units, doc_id_prefix)

def ingest_pdfs_in_directory(directory_path):
    print(f"Collection count before: {collection.count()}")
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                full_path = os.path.join(root, file)
                print(f"Ingesting {full_path} ...")
                ingest_pdf_to_chroma(full_path)
    print(f"Collection count after: {collection.count()}")

if __name__ == "__main__":
    ingest_pdfs_in_directory(SOURCES_DIR)
