from collections import defaultdict

def format_context(docs):
    # Group documents by docid
    docs_by_id = defaultdict(list)
    for doc in docs:
        metadata = doc.get("metadata", {})
        # If metadata is a list, pick the first dict element, else use as is
        if isinstance(metadata, list):
            metadata = metadata[0] if metadata else {}
        docid = metadata.get("docid", "unknown_doc")
        texts = doc.get("text")
        # text might be a list or str; unify it to str
        if isinstance(texts, list):
            text = "\n".join(texts)
        else:
            text = texts or ""
        docs_by_id[docid].append(text)

    # Build context blocks per document
    context_blocks = []
    for docid, texts in docs_by_id.items():
        combined_text = "\n\n".join(texts)
        context_blocks.append(f"Document: {docid}\n{combined_text}")
    final_context = "\n\n---\n\n".join(context_blocks)

    return final_context

def extract_doc_names(retrieved_docs: list) -> list:
    """
    Extracts a list of unique document names from retrieved chunks metadata.
    
    Args:
        retrieved_docs (list): List of chunks where each chunk is a dict containing 'metadata' with 'docid'.
        
    Returns:
        list: Unique document names extracted from 'docid' fields.
    """
    doc_names = set()
    
    for chunk in retrieved_docs:
        # Each chunk may have multiple metadata entries, iterate over them if list
        metadata_list = chunk.get('metadata', [])
        if not isinstance(metadata_list, list):
            metadata_list = [metadata_list]
        for meta in metadata_list:
            docid = meta.get('docid')
            if docid:
                doc_names.add(docid)
    
    print(f"doc_names:\n{doc_names}")

    return list(doc_names)

def build_augmented_prompt(user_question: str, context_docs: str) -> str:
    if not context_docs:
        return user_question
    return (
        "You are a helpful assistant. Use the provided context to focus your answer.\n\n"
        f"Context:\n{context_docs}\n\n"
        f"Question: {user_question}\n"
        "At the end of the reply always provide citations of the source document names in your answer, referencing the given context.\n"
         "Answer:"
    )
