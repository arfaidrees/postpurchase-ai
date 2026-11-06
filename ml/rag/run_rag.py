
from ml.rag.retriever import PolicyRetriever
from ml.rag.context_builder import build_context
from ml.rag.generator import PolicyAnswerGenerator


def main():
    retriever = PolicyRetriever()
    generator = PolicyAnswerGenerator()

    questions = [
        "What should I do if my item arrives damaged?",
        "How many days do I have to return an unused product?",
        "Can I cancel my order after it has shipped?",
    ]

    for question in questions:
        print("\n" + "=" * 60)
        print("Customer:", question)

        documents = retriever.retrieve(
            question=question,
            top_k=2
        )

        context = build_context(documents)

        print("\nRetrieved sources:")
        for document in documents:
            print(
                f"- [{document['id']}] "
                f"{document['title']}"
            )

        answer = generator.generate(
            question=question,
            context=context
        )

        print("\nAI answer:")
        print(answer)


if __name__ == "__main__":
    main()
