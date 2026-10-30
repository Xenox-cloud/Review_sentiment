import pytest

from app.model import SentimentModel


@pytest.fixture(scope="module")
def model():
    m = SentimentModel()
    m.metrics = m.train()
    return m


def test_training_reports_accuracy(model):
    assert model.metrics["rows"] > 50
    assert model.metrics["test_accuracy"] >= 0.7


@pytest.mark.parametrize("text,expected", [
    ("Excellent quality, I love it", "positive"),
    ("Terrible, broke after one day, waste of money", "negative"),
])
def test_clear_reviews(model, text, expected):
    assert model.predict(text).label == expected


def test_confidence_is_probability(model):
    p = model.predict("good product")
    assert 0.0 <= p.confidence <= 1.0


def test_untrained_model_raises():
    with pytest.raises(RuntimeError):
        SentimentModel().predict("hello")
