# Review Sentiment API (FastAPI + scikit-learn + MongoDB)

A FastAPI web app that labels customer reviews as positive, negative or neutral with a scikit-learn model, saves every result in MongoDB and shows them on a small HTML/JavaScript page.

## What it shows

- **Machine learning in a web app:** TF-IDF (character n-grams) + Logistic Regression pipeline, trained from `data/reviews.csv`, saved with joblib and loaded at startup.
- **FastAPI:** REST endpoints with Pydantic validation, dependency injection and app lifespan startup.
- **MongoDB:** `ReviewStore` class with insert, sorted queries, an index and an aggregation pipeline for the summary.
- **Frontend:** plain JavaScript `fetch` calls, live summary and history, safe HTML escaping.
- **Testing + CI/CD:** 12 pytest tests (model + API) using `mongomock`, GitHub Actions for tests and Docker build.
- **Docker:** the image trains the model at build time; `docker-compose.yml` adds MongoDB.

## API

| Method | Path | Description |
|---|---|---|
| POST | `/api/analyze` | Body `{"text": "..."}` → label, confidence, saved record |
| GET | `/api/reviews?limit=20` | Recent analysed reviews |
| GET | `/api/summary` | Counts per label |
| GET | `/health` | Health check |

Interactive docs: http://localhost:8000/docs

## Run with Docker

```bash
docker compose up --build
```

Open http://localhost:8000

## Run without Docker

Needs MongoDB running on `localhost:27017` (or set `MONGO_URI`, for example a MongoDB Atlas URL).

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python -m app.model          # train and save the model
uvicorn app.main:app --reload
```

## Tests

```bash
pytest -q
```

## Notes

The dataset is small (133 hand-written reviews), so held-out accuracy is about 70%. Results under 60% confidence are shown as "neutral". More data would raise accuracy; the pipeline does not need to change.
