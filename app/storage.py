"""MongoDB storage for analysed reviews."""
from datetime import datetime, timezone

from pymongo import DESCENDING, MongoClient
from pymongo.collection import Collection


class ReviewStore:
    def __init__(self, collection: Collection):
        self.collection = collection
        self.collection.create_index([("created_at", DESCENDING)])

    @classmethod
    def from_uri(cls, uri: str, db_name: str = "sentiment") -> "ReviewStore":
        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        return cls(client[db_name]["reviews"])

    def save(self, text: str, label: str, confidence: float) -> dict:
        doc = {
            "text": text,
            "label": label,
            "confidence": confidence,
            "created_at": datetime.now(timezone.utc),
        }
        result = self.collection.insert_one(doc)
        return self._serialize({**doc, "_id": result.inserted_id})

    def recent(self, limit: int = 20) -> list[dict]:
        cursor = self.collection.find().sort("created_at", DESCENDING).limit(limit)
        return [self._serialize(d) for d in cursor]

    def summary(self) -> dict:
        pipeline = [{"$group": {"_id": "$label", "count": {"$sum": 1}}}]
        counts = {row["_id"]: row["count"] for row in self.collection.aggregate(pipeline)}
        return {
            "total": sum(counts.values()),
            "positive": counts.get("positive", 0),
            "negative": counts.get("negative", 0),
            "neutral": counts.get("neutral", 0),
        }

    @staticmethod
    def _serialize(doc: dict) -> dict:
        return {
            "id": str(doc["_id"]),
            "text": doc["text"],
            "label": doc["label"],
            "confidence": doc["confidence"],
            "created_at": doc["created_at"].isoformat(),
        }
