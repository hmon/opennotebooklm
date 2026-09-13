from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import db
from app.ingest import ingest
from app.pipeline import answer_question

STATIC = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    yield


app = FastAPI(title="opennotebooklm", lifespan=lifespan)


class CorpusIn(BaseModel):
    name: str


class QueryIn(BaseModel):
    question: str
    mode: str = "synthesis"


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/corpora")
def create_corpus(body: CorpusIn):
    row = db.fetchone(
        "INSERT INTO corpora (name) VALUES (%s)"
        " ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id, name",
        (body.name,),
    )
    return row


@app.get("/corpora")
def list_corpora():
    return db.fetchall("SELECT id, name, created_at FROM corpora ORDER BY id")


@app.get("/corpora/{corpus_id}/documents")
def list_documents(corpus_id: int):
    return db.fetchall(
        "SELECT d.id, d.title, d.source_type, d.file_name, d.version, d.ingested_at,"
        " (SELECT count(*) FROM chunks c WHERE c.document_id = d.id AND c.version = d.version)"
        " AS chunks"
        " FROM documents d WHERE d.corpus_id = %s ORDER BY d.ingested_at",
        (corpus_id,),
    )


@app.post("/corpora/{corpus_id}/documents")
async def upload_document(
    corpus_id: int,
    file: UploadFile = File(...),
    title: str | None = Form(None),
):
    if not db.fetchone("SELECT id FROM corpora WHERE id = %s", (corpus_id,)):
        raise HTTPException(404, "corpus not found")
    data = await file.read()
    try:
        return ingest(corpus_id, file.filename or "upload", data, title)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/corpora/{corpus_id}/query")
def query(corpus_id: int, body: QueryIn):
    if not db.fetchone("SELECT id FROM corpora WHERE id = %s", (corpus_id,)):
        raise HTTPException(404, "corpus not found")
    if not body.question.strip():
        raise HTTPException(400, "empty question")
    return answer_question(corpus_id, body.question, body.mode)


@app.get("/chunks/{chunk_id}")
def get_chunk(chunk_id: str):
    """Resolve a citation to its exact source passage."""
    row = db.fetchone(
        "SELECT c.id AS chunk_id, c.document_id, c.version, c.page, c.section,"
        " c.start_char, c.end_char, c.text, d.title, d.file_name, d.source_type"
        " FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.id = %s",
        (chunk_id,),
    )
    if not row:
        raise HTTPException(404, "chunk not found")
    return row
