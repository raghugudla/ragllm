# 1. Document Parsing (Text, Images, Tables, Math)
from pathlib import Path
import re
from typing import List, Dict, Any

import networkx as nx
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

def ingest_document(document_path, embedding_model):
    """
    Parse document, build graphs, embed nodes, and store embeddings in vector index.
    Returns the updated vector index or any summary info if needed.
    """
    parsed = parse_document(document_path)
    graph = build_cross_modal_graph(parsed)
    vector_index = embed_graph_nodes(graph, embedding_model)
    # Optionally store or return vector_index depending on design
    return vector_index

def parse_document(document_path):
    text_chunks = parse_text(document_path)             # text entities
    images = extract_images(document_path)              # image entities with metadata
    tables = parse_tables(document_path)                # structured table entities
    math_expressions = extract_math_expressions(document_path)  # math entities
    return {
        "text": text_chunks,
        "images": images,
        "tables": tables,
        "math": math_expressions,
    }


# ---- 1.a Text parsing -----------------------------------------------------

def _build_text_entity(entity_id: str, content: str) -> Dict[str, Any]:
    content = content.strip()
    if not content:
        return {}
    return {
        "id": entity_id,
        "content": content,
        "modality": "text",
        # Local summary as in RAG-Anything entity design
        "entity_summary": content[:300],
    }


def parse_text(document_path: str) -> List[Dict[str, Any]]:
    """
    Decompose document text into fine-grained entities, following RAG-Anything's
    idea of small, semantically coherent units:

    - PDFs: one text entity per page (can later be refined to paragraph/section).
    - Other text-like files: a single entity for the whole file.
    """
    path = Path(document_path)
    ext = path.suffix.lower()

    entities: List[Dict[str, Any]] = []

    if ext == ".pdf":
        try:
            reader = PdfReader(str(path))
        except Exception:
            return []

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            entity = _build_text_entity(f"text_page{i}", text)
            if entity:
                entities.append(entity)
    else:
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []

        entity = _build_text_entity("text_file0", raw)
        if entity:
            entities.append(entity)

    return entities


# ---- 1.b Image parsing ----------------------------------------------------

def extract_images(document_path: str) -> List[Dict[str, Any]]:
    """
    Extract image entities.

    In the full RAG-Anything framework, this would handle:
    - Embedded figures in PDFs.
    - Cross-modal links between text and images.

    Here we implement a minimal version:
    - If the document itself is an image file (png/jpg/jpeg/webp/gif), return a
      single image entity pointing to that file.
    """
    path = Path(document_path)
    ext = path.suffix.lower()

    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    entities: List[Dict[str, Any]] = []

    if ext in image_exts:
        entities.append(
            {
                "id": "image_0",
                "content": str(path),
                "modality": "image",
                "entity_summary": f"Image file at {path.name}",
            }
        )

    # Placeholder: no embedded PDF image extraction yet
    return entities


# ---- 1.c Table parsing ----------------------------------------------------

def parse_tables(document_path: str) -> List[Dict[str, Any]]:
    """
    Extract table entities.

    The RAG-Anything framework uses layout-aware parsing and normalization of
    tables into structured entities. That is out of scope here, so we keep a
    placeholder that returns an empty list while preserving the API.
    """
    return []


# ---- 1.d Math expression parsing -----------------------------------------

def extract_math_expressions(document_path: str) -> List[Dict[str, Any]]:
    """
    Extract math entities from text using simple LaTeX-style pattern matching,
    echoing RAG-Anything's idea of treating equations as separate entities.

    Looks for:
    - Inline math: $...$
    - Display math: \\[ ... \\]
    """
    text_entities = parse_text(document_path)
    if not text_entities:
        return []

    combined_text = "\n\n".join(e["content"] for e in text_entities)

    patterns = [
        r"\$(.+?)\$",          # inline math
        r"\\\[(.+?)\\\]",      # display math
    ]

    matches: List[str] = []
    for pat in patterns:
        matches.extend(re.findall(pat, combined_text, flags=re.DOTALL))

    entities: List[Dict[str, Any]] = []
    for i, expr in enumerate(matches):
        expr_clean = expr.strip()
        if not expr_clean:
            continue
        entities.append(
            {
                "id": f"math_{i}",
                "content": expr_clean,
                "modality": "math",
                "entity_summary": expr_clean[:200],
            }
        )

    return entities

# 2. Dual-Graph Construction
def build_cross_modal_graph(parsed_data):
    # Create nodes for each modality: text entities, image captions, table headers, math symbols
    graph = Graph()
    graph.add_nodes_from_text(parsed_data["text"])
    graph.add_nodes_from_images(parsed_data["images"])
    graph.add_nodes_from_tables(parsed_data["tables"])
    graph.add_nodes_from_math(parsed_data["math"])
    # Add relations and align entities across modalities
    graph.align_entities()
    return graph

# 3. Embeddings and Vector Index
def embed_graph_nodes(graph, embedding_model):
    for node in graph.nodes:
        node.embedding = embedding_model.encode(node.content)
    vector_index = VectorStore()
    vector_index.add_embeddings([(node.id, node.embedding) for node in graph.nodes])
    return vector_index

# 4. Hybrid Retrieval
def hybrid_retrieve(query, graph, vector_index, query_embedding_model):
    query_embedding = query_embedding_model.encode(query)
    # Semantic search via vector index
    semantic_matches = vector_index.search(query_embedding)
    # Structural search via graph traversal
    structural_matches = graph.traverse_relevant_nodes(query)
    # Combine and rerank matches
    combined_results = rank_combine(semantic_matches, structural_matches, query)
    return combined_results

# 5. Synthesis with Vision-Language Models
def generate_response(query, retrieval_results, vlm_model):
    # Prepare multimodal input: text context + images/tables/math visualizations
    context_text = compose_textual_context(retrieval_results)
    related_visuals = collect_visuals(retrieval_results)
    # VLM generates grounded response
    response = vlm_model.generate(query, context_text, related_visuals)
    return response

# Example main flow
def rag_anything_pipeline(document_path, query):
    parsed = parse_document(document_path)
    graph = build_cross_modal_graph(parsed)
    embedding_model = load_embedding_model()
    vector_index = embed_graph_nodes(graph, embedding_model)
    query_embedding_model = load_query_embedding_model()
    retrieval_results = hybrid_retrieve(query, graph, vector_index, query_embedding_model)
    vlm_model = load_vision_language_model()
    response = generate_response(query, retrieval_results, vlm_model)
    return response


