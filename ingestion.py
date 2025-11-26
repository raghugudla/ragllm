import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

from rag_anything.anything import parse_document

PERSIST_DIR = "./data/chroma_db"
SOURCES_DIR = "./data/sources"
client = chromadb.PersistentClient(path=PERSIST_DIR)

collection = client.get_or_create_collection("pdf_collection")
model = SentenceTransformer("all-MiniLM-L6-v2")

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


def extract_multimodal_units(file_path: str):
    """
    Multimodal extractor aligned with the RAG-ANYTHING framework implemented in
    `rag_anything.anything.parse_document`.

    `parse_document` returns modality-specific entities:
      { "text": [...], "images": [...], "tables": [...], "math": [...] }

    Here we flatten them into generic units that `ingest_units_to_chroma` can store
    in Chroma.
    """
    parsed = parse_document(file_path)
    units = []

    for modality, entities in parsed.items():
        for idx, entity in enumerate(entities):
            unit_id = entity.get("id") or f"{modality}_{idx}"
            content = entity.get("content", "")
            if not content:
                continue

            units.append(
                {
                    "id": unit_id,
                    "content": content,
                    "modality": entity.get("modality", modality),
                    "entity_summary": entity.get("entity_summary"),
                }
            )

    return units


def get_doc_title_version(doc_path: str):
    """
    Build a stable doc id prefix from metadata + mtime.
    - PDFs: use PDF title when available, else filename stem.
    - Others: use filename stem.
    """
    path = Path(doc_path)
    ext = path.suffix.lower()
    title = path.stem

    if ext == ".pdf":
        try:
            reader = PdfReader(str(path))
            meta = reader.metadata  # new API
            if meta and getattr(meta, "title", None):
                title = meta.title
        except Exception as e:
            print(f"Warning: could not read PDF metadata for {doc_path}: {e}")

    mod_time = os.path.getmtime(doc_path)
    version = str(int(mod_time))
    title = title.replace(" ", "_")
    return title, version


def ingest_doc_to_chroma(doc_path: str):
    title, version = get_doc_title_version(doc_path)
    doc_id_prefix = f"{title}_v{version}"
    units = extract_multimodal_units(doc_path)
    ingest_units_to_chroma(units, doc_id_prefix)


def ingest_files_in_directory(directory_path: str):
    print(f"Collection count before: {collection.count()}")
    for root, _, files in os.walk(directory_path):
        for file in files:
            full_path = os.path.join(root, file)
            print(f"Ingesting {full_path} ...")
            ingest_doc_to_chroma(full_path)
    print(f"Collection count after: {collection.count()}")


if __name__ == "__main__":
    ingest_files_in_directory(SOURCES_DIR)
