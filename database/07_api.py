"""
Step 7: FastAPI backend.

Loads the embedding model, Milvus connection, and BM25 index ONCE at
startup, then serves /chat requests using the same retrieval+generation
pipeline built in Step 6.

Run with:
    pip install fastapi uvicorn
    uvicorn 07_api:app --reload --port 8000

Then test with:
    curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"query\": \"your question\"}"

Note: this file imports functions from 06_hybrid_retrieve_and_generate.py,
so keep both files (and law_project.db, bm25_index.pkl, bm25_chunks.pkl) in
the same folder.
"""

import importlib.util
import sys
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import Step 6 as a module (its filename starts with a digit, so a normal
# `import` statement won't work - load it explicitly instead).
_step6_path = Path(__file__).parent / "06_hybrid_retrieve_and_generate.py"
_spec = importlib.util.spec_from_file_location("step6", _step6_path)
step6 = importlib.util.module_from_spec(_spec)
sys.modules["step6"] = step6
_spec.loader.exec_module(step6)

app = FastAPI(title="Corporate Law RAG Chatbot")

# Local portfolio project - the frontend is a static HTML file opened
# directly in the browser, so its origin isn't http://localhost. Allowing
# all origins is fine for local personal use; tighten this if you ever
# deploy this somewhere other users can reach.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

resources = {}


class ChatRequest(BaseModel):
    query: str


class SourceItem(BaseModel):
    label: int
    citation: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


@app.on_event("startup")
def startup_event():
    print("Loading embedding model, Milvus collection, and BM25 index...")
    embed_model, milvus_client, bm25, bm25_chunks = step6.load_resources()
    resources["embed_model"] = embed_model
    resources["milvus_client"] = milvus_client
    resources["bm25"] = bm25
    resources["bm25_chunks"] = bm25_chunks
    print("Ready.")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = step6.ask(
        req.query,
        resources["embed_model"],
        resources["milvus_client"],
        resources["bm25"],
        resources["bm25_chunks"],
    )
    return result


@app.get("/health")
def health():
    return {"status": "ok", "loaded": bool(resources)}
@app.get("/")
def home():
    return FileResponse(Path(__file__).parent.parent / "frontend.html")