import chromadb
from sentence_transformers import SentenceTransformer


PERSIST_DIR = "./data/chroma_db"
COLLECTION_NAME = "pdf_collection"

_client = chromadb.PersistentClient(path=PERSIST_DIR)
_collection = _client.get_or_create_collection(COLLECTION_NAME)
_embedder = SentenceTransformer('all-MiniLM-L6-v2')


def retrieve_context(query: str, k: int = 4) -> list[str]:
    try:
        if _collection.count() == 0:
            return []
        q_emb = _embedder.encode([query]).tolist()
        res = _collection.query(
            query_embeddings=q_emb,
            n_results=k,
            include=["documents"],
        )
        return (res.get("documents") or [[]])[0]
    except Exception:
        return []


def collection_count() -> int:
    try:
        return _collection.count()
    except Exception:
        return 0


