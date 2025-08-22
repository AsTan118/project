from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import hashlib
import pdfplumber


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma


from .settings import VECTORDB_DIR, CONFIG

def _hash_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]

def extract_pages(pdf_path: Path) -> List[Dict]:
    data: List[Dict] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                data.append({"page": i + 1, "content": text})
    return data


def to_documents(pages: List[Dict]) -> List[Document]:
    full_text = "\n\n".join([p["content"] for p in pages])
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=120, separators=["\n\n", "\n", ". ", " "])
    chunks = splitter.split_text(full_text)
    docs = [
        Document(page_content=ch, metadata={"source": "uploaded_pdf"}) for ch in chunks
    ]
    return docs


def build_or_load_vectordb(collection_id: str, docs: List[Document] | None = None):
    """Create a Chroma collection (if docs given) or load an existing one."""
    embeddings = OllamaEmbeddings(model=CONFIG.embed_model)
    persist_dir = VECTORDB_DIR / collection_id


    if docs is not None:
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=str(persist_dir),
        )
        vectordb.persist()
        return vectordb
    # If no docs provided, load existing collection
    return Chroma(
        persist_directory=str(persist_dir), embedding_function=embeddings
    )



def ingest_pdf(pdf_path: Path) -> Dict:
    pages = extract_pages(pdf_path)
    docs = to_documents(pages)
    # stable id derived from file content, so re-uploads reuse same collection
    collection_id = f"{CONFIG.collection_prefix}{_hash_file(pdf_path)}"
    build_or_load_vectordb(collection_id, docs)
    return {
        "collection_id": collection_id,
        "num_pages": len(pages),
        "num_chunks": len(docs),
    }



def retrieve(collection_id: str, query: str, k: int = 4) -> List[Document]:
    vectordb = build_or_load_vectordb(collection_id)
    return vectordb.similarity_search(query, k=k)


def generate_answer(context_docs: List[Document], question: str) -> str:
    context = "\n\n".join([d.page_content for d in context_docs])
    system_prompt = (
        "You are a helpful assistant. Answer the user's question using the given context. "
        "If the answer isn't in the context, say you don't know and suggest where to look."
    )
    prompt = f"""SYSTEM:\n{system_prompt}\n\nCONTEXT:\n{context}\n\nUSER QUESTION:\n{question}\n\nASSISTANT:"""


    chat = ChatOllama(model=CONFIG.chat_model, temperature=0.1)
    resp = chat.invoke(prompt)
    return resp.content if hasattr(resp, "content") else str(resp)