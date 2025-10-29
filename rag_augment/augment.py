def format_context(docs: list[str]) -> str:
    if not docs:
        return ""
    parts = []
    for d in docs:
        text = (d or "").strip()
        if len(text) > 2000:
            text = text[:2000] + "\n..."
        parts.append(text)
    return "\n\n---\n\n".join(parts)


def build_augmented_prompt(user_question: str, context: str) -> str:
    if not context:
        return user_question
    return (
        "You are a helpful assistant. Use the provided context to answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_question}\n"
        "Answer:"
    )


