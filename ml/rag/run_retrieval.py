
from ml.rag.retriever import PolicyRetriever
from ml.rag.context_builder import build_context


def main():
    retriever = PolicyRetriever()

    questions = [
        "Can I return shoes after 20 days?",
        "How long does a refund take?",
        "Can I cancel an order after shipping?",
        "What should I do if my item is damaged?"
    ]

    for question in questions:
        print("\n" + "=" * 60)
        print("Customer question:", question)

        documents = retriever.retrieve(
            question=question,
            top_k=2
        )

        print("\nRetrieved documents:")

        for document in documents:
            print(
                f"- {document['title']} "
                f"(distance={document['distance']})"
            )

        context = build_context(documents)

        print("\nPrepared LLM context:")
        print(context)


if __name__ == "__main__":
    main()
