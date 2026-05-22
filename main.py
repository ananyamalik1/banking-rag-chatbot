import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from rag import RAGPipeline

app = FastAPI(title="Banking Support Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGPipeline()
sessions = {}  # session_id -> chat history

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    sources: list[str] = []

@app.get("/health")
def health():
    return {"status": "ok", "docs_loaded": rag.get_doc_count()}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(400, "Only PDF and TXT files are supported.")
    content = await file.read()
    result = rag.ingest(content, file.filename)
    return {"message": f"Ingested '{file.filename}'", "chunks": result["chunks"]}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    history = sessions.get(session_id, [])

    reply, sources = rag.query(req.message, history)

    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": reply})
    sessions[session_id] = history[-10:]  # keep last 5 turns

    return ChatResponse(reply=reply, session_id=session_id, sources=sources)