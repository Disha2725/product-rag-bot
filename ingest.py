import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


def load_excel_documents(data_dir: Path):
    docs = []
    print(f"[DEBUG] Scanning for Excel files under: {data_dir}")
    for path in data_dir.glob("**/*.xlsx"):
        print(f"[DEBUG] Found Excel file: {path}")
        try:
            df = pd.read_excel(path)
            print(f"[DEBUG] {path.name}: {len(df)} rows")
        except Exception as e:
            print(f"[WARN] Failed to read {path}: {e}")
            continue

        for idx, row in df.iterrows():
            # Build a text blob from all non-null columns in the row
            parts = []
            for col, val in row.items():
                if pd.isna(val):
                    continue
                parts.append(f"{col}: {val}")
            if not parts:
                continue

            text = "\n".join(parts)
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(path),
                        "row_index": int(idx),
                    },
                )
            )
    
    print(f"[DEBUG] Total docs created from Excel: {len(docs)}")
    return docs


def main():
    load_dotenv()

    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    persist_dir = base_dir / "chroma_db"
    persist_dir.mkdir(exist_ok=True)

    print(f"Loading product Excel documents from {data_dir}...")
    docs = load_excel_documents(data_dir)
    print(f"Loaded {len(docs)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    split_docs = splitter.split_documents(docs)
    print(f"Split into {len(split_docs)} chunks")

    if not split_docs:
        print("No documents to index; skipping vector store build.")
        return

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Building and persisting Chroma vector store for product docs...")
    Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )

    print("Done. Vector store saved to:", persist_dir)


if __name__ == "__main__":
    main()
