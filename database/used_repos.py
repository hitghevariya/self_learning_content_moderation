from pymongo import MongoClient
from datetime import datetime
import os
from config import DB_NAME,MONGO

client = MongoClient(MONGO)
db = client[DB_NAME]

used_repos = db.used_repos


def is_repo_checked(repo_url: str) -> bool:
    return used_repos.find_one({"repo_url": repo_url}) is not None


def mark_repo_checked(repo_url: str, status: str, reason: str = None):
    used_repos.update_one(
        {"repo_url": repo_url},
        {
            "$set": {
                "status": status,      # "parsed" | "failed"
                "reason": reason,
                "checked_at": datetime.utcnow()
            }
        },
        upsert=True
    )
