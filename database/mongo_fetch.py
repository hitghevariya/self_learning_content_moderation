from pymongo import MongoClient
from config import MONGO

def fetch_captions_from_mongo():
    client = MongoClient(MONGO)
    db = client.photoApp
    collection = db.posts   

    captions = []
    for doc in collection.find({}, {"_id": 0, "caption": 1}):
        if doc.get("caption"):
            captions.append(doc["caption"])

    return captions
