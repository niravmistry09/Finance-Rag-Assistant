from contextlib import asynccontextmanager
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_app.generation import RAGService


service = RAGService()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        yield
    finally:
        service.close()


app = FastAPI(title="Finance RAG API", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    contexts: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        result = service.answer(request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(
        answer=result.answer,
        sources=result.sources,
        contexts=result.contexts,
    )
