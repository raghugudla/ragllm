def format_context(docs: list[dict]) -> str:
    if not docs:
        return ""

    parts = []
    for d in docs:
        text = (d.get("text") or "").strip()
        metadata = d.get("metadata") or {}

        if len(text) > 2000:
            text = text[:2000] + "\n..."

        citation = ""
        if "source" in metadata:
            citation = f" [source: {metadata['source']}]"
        elif "filename" in metadata:
            citation = f" [source: {metadata['filename']}]"
        print(f"Citation: {citation}")

        parts.append(text + citation)

    return "\n\n---\n\n".join(parts)


def build_augmented_prompt(user_question: str, context: str) -> str:
    if not context:
        return user_question

    return (
        "You are a helpful assistant. Use the provided context to answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_question}\n"
        "Please provide citations in the answer referencing the given sources.\n"
        "Answer:"
    )
