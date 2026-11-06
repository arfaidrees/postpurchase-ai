
def build_context(documents: list[dict]) -> str:
    if not documents:
        return "No relevant policy documents were retrieved."

    sections = []

    for document in documents:
        sections.append(
            f"[Source: {document['id']}]\n"
            f"Title: {document['title']}\n"
            f"Content: {document['content']}"
        )

    return "\n\n---\n\n".join(sections)
