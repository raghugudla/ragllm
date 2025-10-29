import os
import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# New way to initialize the client and specify persistence
PERSIST_DIR = "./data/chroma_db"
SOURCES_DIR = "./data/sources"
client = chromadb.PersistentClient(path=PERSIST_DIR)

# Create or get your collection
collection = client.get_or_create_collection("pdf_collection")

# Load SentenceTransformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + " "
    return text.strip()

def ingest_pdf_to_chroma(pdf_path):
    document_text = extract_text_from_pdf(pdf_path)
    embeddings = model.encode([document_text])
    doc_id = pdf_path

    existing = collection.get(ids=[doc_id])
    print(f"Existing entries for {doc_id}: {existing['ids']}")

    if existing.get("ids"):
        print(f"Document {doc_id} already ingested. Skipping.")
        return

    collection.upsert(
        documents=[document_text],
        embeddings=embeddings.tolist(),
        ids=[doc_id]
    )
    print(f"Ingested {doc_id}")

def ingest_user_uploaded_pdf(file_path):
    ingest_pdf_to_chroma(file_path)
    # Persist changes
    #client.persist()
    print(f"Successfully ingested user file: {file_path}")


def ingest_pdfs_in_directory(directory_path):
    print(f"Collection count before: {collection.count()}")
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                full_path = os.path.join(root, file)
                print(f"Ingesting {full_path} ...")
                ingest_pdf_to_chroma(full_path)
    print(f"Collection count after: {collection.count()}")

#ingest_pdfs_in_directory("data/sources")

if __name__ == "__main__":
    # Ingest all PDFs under data/sources
    ingest_pdfs_in_directory(SOURCES_DIR)