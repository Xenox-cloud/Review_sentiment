import mongomock
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model import SentimentModel
from app.storage import ReviewStore


@pytest.fixture(scope="module")
def trained_model():
    m = SentimentModel()
    m.train()
    return m


@pytest.fixture
def client(trained_model):
    # mongomock gives an in-memory MongoDB, so tests need no real server.
    app.state.model = trained_model
    app.state.store = ReviewStore(mongomock.MongoClient()["test"]["reviews"])
    with TestClient(app) as c:
        yield c
    del app.state.model, app.state.store


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_analyze_saves_review(client):
    res = client.post("/api/analyze", json={"text": "Amazing product, highly recommend"})
    assert res.status_code == 201
    body = res.json()
    assert body["label"] == "positive"
    assert body["id"]

    history = client.get("/api/reviews").json()
    assert len(history) == 1
    assert history[0]["text"] == "Amazing product, highly recommend"


@pytest.mark.parametrize("payload", [{}, {"text": "   "}, {"text": "x" * 1001}])
def test_analyze_rejects_bad_input(client, payload):
    res = client.post("/api/analyze", json=payload)
    assert res.status_code == 422


def test_summary_counts_labels(client):
    client.post("/api/analyze", json={"text": "Great quality, love it"})
    client.post("/api/analyze", json={"text": "Awful, broke after a week, refund please"})
    summary = client.get("/api/summary").json()
    assert summary["total"] == 2
    assert summary["positive"] + summary["negative"] + summary["neutral"] == 2


def test_index_page_loads(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Review Sentiment Analyzer" in res.text
