# query.py
from rag import load_vectorstore, build_qa_chain, format_docs

def main():
    vectordb = load_vectorstore()
    chain, retriever = build_qa_chain(vectordb)

    print("RAG ready. Type your question (or 'exit').")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in {"exit", "quit"}:
            break

        answer = chain.invoke(q)
        print("\n=== Answer ===")
        print(answer)

        print("\n=== Sources ===")
        for d in retriever.invoke(q):
            src = d.metadata.get("source", "?")
            page = d.metadata.get("page", "")
            print(f"- {src}" + (f" (page {page})" if page != "" else ""))

if __name__ == "__main__":
    main()