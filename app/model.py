"""Text sentiment model: TF-IDF features + Logistic Regression (scikit-learn)."""
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "reviews.csv"
MODEL_PATH = ROOT / "models" / "sentiment.joblib"


@dataclass
class Prediction:
    label: str
    confidence: float

    def to_dict(self) -> dict:
        return {"label": self.label, "confidence": round(self.confidence, 3)}


class SentimentModel:
    """Wraps training, saving, loading and prediction behind one small class."""

    # Below this confidence we call the result "neutral" instead of guessing.
    NEUTRAL_THRESHOLD = 0.6

    def __init__(self, pipeline: Pipeline | None = None):
        self.pipeline = pipeline

    @staticmethod
    def build_pipeline() -> Pipeline:
        # Character n-grams cope better than whole words with a small dataset
        # ("broke" still matches "broken", typos still match).
        return Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, C=10.0)),
        ])

    def train(self, data_path: Path = DATA_PATH) -> dict:
        df = pd.read_csv(data_path).dropna()
        x_train, x_test, y_train, y_test = train_test_split(
            df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
        )
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(x_train, y_train)
        test_accuracy = accuracy_score(y_test, self.pipeline.predict(x_test))
        # Refit on all rows so the saved model uses every example.
        self.pipeline.fit(df["text"], df["label"])
        return {"rows": len(df), "test_accuracy": round(float(test_accuracy), 3)}

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load_or_train(cls, path: Path = MODEL_PATH) -> "SentimentModel":
        if path.exists():
            return cls(joblib.load(path))
        model = cls()
        model.train()
        model.save(path)
        return model

    def predict(self, text: str) -> Prediction:
        if self.pipeline is None:
            raise RuntimeError("Model is not trained")
        probs = self.pipeline.predict_proba([text])[0]
        best = probs.argmax()
        confidence = float(probs[best])
        label = str(self.pipeline.classes_[best])
        if confidence < self.NEUTRAL_THRESHOLD:
            label = "neutral"
        return Prediction(label=label, confidence=confidence)


if __name__ == "__main__":
    m = SentimentModel()
    print("Training:", m.train())
    m.save()
    print("Saved to", MODEL_PATH)
