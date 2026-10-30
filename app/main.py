import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .model import SentimentModel
from .storage import ReviewStore

STATIC_DIR = Path(__file__).parent / "static"


class ReviewIn(BaseModel):
    text: str = Field(max_length=1000)

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty")
        return v


class ReviewOut(BaseModel):
    id: str
    text: str
    label: str
    confidence: float
    created_at: str


class Summary(BaseModel):
    total: int
    positive: int
    negative: int
    neutral: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tests set these on app.state before startup; otherwise load the real ones.
    if not hasattr(app.state, "model"):
        app.state.model = SentimentModel.load_or_train()
    if not hasattr(app.state, "store"):
        app.state.store = ReviewStore.from_uri(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    yield


app = FastAPI(title="Review Sentiment API", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def get_model(request: Request) -> SentimentModel:
    return request.app.state.model


def get_store(request: Request) -> ReviewStore:
    return request.app.state.store


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze", response_model=ReviewOut, status_code=201)
def analyze(review: ReviewIn, model: SentimentModel = Depends(get_model),
            store: ReviewStore = Depends(get_store)):
    prediction = model.predict(review.text)
    return store.save(review.text, prediction.label, prediction.to_dict()["confidence"])


@app.get("/api/reviews", response_model=list[ReviewOut])
def reviews(limit: int = Query(default=20, ge=1, le=100), store: ReviewStore = Depends(get_store)):
    return store.recent(limit)


@app.get("/api/summary", response_model=Summary)
def summary(store: ReviewStore = Depends(get_store)):
    return store.summary()
