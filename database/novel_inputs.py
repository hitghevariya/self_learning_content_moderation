from pymongo import MongoClient
from datetime import datetime


from config import DB_NAME,MONGO

# Mongo connection
MONGO_URI = MONGO
if not MONGO_URI:
    raise RuntimeError(" MONGO_URI not set in environment")


client = MongoClient(MONGO_URI)
db = client[DB_NAME]

novel_inputs = db.novel_inputs


def store_novel_input(features: dict, decision: str, score: float):
    print(" store_novel_input CALLED")
    print("Decision:", decision, "Score:", score)

    novel_inputs.insert_one({
        "features": features,
        "decision": decision,
        "score": float(score),
        "used": False,
        "createdAt": datetime.utcnow()
    })

    print("Insert attempted")

def fetch_unused_novel_tokens(limit=50):
    """
    Extract slang / novel tokens from unused OOD captions
    """
    docs = novel_inputs.find(
        {"used": False},
        {"caption": 1}
    ).limit(limit)

    tokens = set()
    for d in docs:
        caption = d.get("caption", "")
        for t in caption.lower().split():
            if len(t) >= 4:
                tokens.add(t)

    return list(tokens)


def mark_novel_inputs_used():
    novel_inputs.update_many(
        {"used": False},
        {"$set": {"used": True}}
    )

def has_unused_novel_inputs():
    """
    Check if there are any unused OOD samples
    """
    return novel_inputs.count_documents({"used": False}) > 0
