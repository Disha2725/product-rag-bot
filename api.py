from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()

base_dir = Path(__file__).parent
persist_dir = base_dir / "chroma_db"

if not persist_dir.exists():
    raise RuntimeError(
        "Vector store not found. Run `python ingest.py` in the project root to build the Chroma DB before starting the API."
    )

embeddings = OpenAIEmbeddings()
vector_store = Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


def retrieve_relevant_chunks(query: str, k: int = 5) -> List[str]:
    docs = vector_store.similarity_search(query, k=k)
    return [d.page_content for d in docs]


def answer_with_context(question: str, context_chunks: List[str]) -> str:
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


app = FastAPI(title="Product RAG Chatbot API")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    chunks = retrieve_relevant_chunks(req.question, k=5)
    answer = answer_with_context(req.question, chunks)
    return ChatResponse(answer=answer)
