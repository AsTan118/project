from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil


from .settings import UPLOAD_DIR
from .rag import ingest_pdf, retrieve, generate_answer


app = FastAPI(title="Local RAG API", version="0.1")


# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")


    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)


    info = ingest_pdf(dest)
    return JSONResponse({"message": "Ingested", **info})


@app.post("/ask")
async def ask_question(payload: dict):
    collection_id = payload.get("collection_id")
    question = payload.get("question")
    k = int(payload.get("k", 4))


    if not collection_id or not question:
        raise HTTPException(status_code=400, detail="collection_id and question are required")


    docs = retrieve(collection_id, question, k=k)
    answer = generate_answer(docs, question)
    sources = [d.metadata for d in docs]


    return {"answer": answer, "sources": sources}