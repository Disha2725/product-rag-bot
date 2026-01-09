import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def load_vector_store(base_dir: Path) -> Chroma:
    persist_dir = base_dir / "chroma_db"
    if not persist_dir.exists():
        raise RuntimeError(
            "Vector store not found. Run `python ingest.py` in the project root to build the Chroma DB."
        )
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)


def retrieve_relevant_chunks(store: Chroma, query: str, k: int = 4) -> List[str]:
    docs = store.similarity_search(query, k=k)
    return [d.page_content for d in docs]


def answer_with_context(llm: ChatOpenAI, question: str, context_chunks: List[str]) -> str:
    context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no retrieved context)"
    system = (
        "You are a helpful product information assistant for internal and external users. "
        "You answer questions based only on the provided context from structured product data (Excel rows) "
        "plus general, non-confidential product knowledge. If the context does not contain the answer, "
        "say you are not sure and suggest checking official product documentation or contacting support. "
        "Do not invent specifications, prices, or commitments that are not clearly stated in the context."
    )
    user_content = (
        f"User question: {question}\n\n"
        f"Context from product knowledge base:\n{context_text}\n\n"
        "Give a concise answer. If helpful, structure it as: (1) direct answer, (2) supporting details from context, "
        "(3) limitations or next steps."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    resp = llm.invoke(messages)
    return resp.content


def main():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Create a .env file with OPENAI_API_KEY=...")

    base_dir = Path(__file__).parent

    print("Loading product vector store...")
    store = load_vector_store(base_dir)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    print("\nProduct RAG Chatbot (PDF/PPT-based)")
    print("Ask any product-related question based on your documents. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        try:
            print("[Chatbot] Retrieving relevant product information...")
            chunks = retrieve_relevant_chunks(store, question, k=5)
            answer = answer_with_context(llm, question, chunks)
            print("\nChatbot:\n" + answer + "\n")
        except Exception as e:
            print(f"[Error] {e}")


if __name__ == "__main__":
    main()
