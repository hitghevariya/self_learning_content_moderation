# from pymongo import MongoClient
# from policy_engine import evaluate_caption
# from datetime import datetime
# import time

# # MongoDB connection
# client = MongoClient("mongodb://127.0.0.1:27017/")
# db = client.photoApp

# queue = db.captionqueues     
# final = db.captions          


# def process_one_caption():
#     # Atomically pick ONE pending job
#     job = queue.find_one_and_update(
#         {"status": "pending"},
#         {"$set": {"status": "processing"}},
#     )

#     if not job:
#         return False  # queue empty

#     caption = job.get("caption", "").strip()

#     # Skip empty captions
#     if not caption:
#         queue.delete_one({"_id": job["_id"]})
#         return True

#     # Run your policy engine
#     result = evaluate_caption(caption)

#     # Save final decision
#     final.insert_one({
#     "caption": caption,
#     "imageId": job["imageId"],
#     "decision": result["decision"],
#     "xgboostScore": result["xgboost_score"],
#     "finalScore": result["final_score"], 
#     "matchedWords": result["matched_words"],
#     "moderatedAt": datetime.utcnow()
# })


#     # Remove from queue
#     queue.delete_one({"_id": job["_id"]})

#     print(f" Moderated → {caption} | {result['decision']}")
#     return True


# if __name__ == "__main__":
#     print(" Queue moderation worker started")

#     while True:
#         worked = process_one_caption()
#         if not worked:
#             time.sleep(10)
# from dotenv import load_dotenv
import os  
from pymongo import MongoClient
from datetime import datetime
from prediction.predict_caption import predict_caption
from config import MONGO

client = MongoClient(MONGO)
db = client.photoApp

queue = db.captionqueues
final = db.captions

print("Queue worker started")

while True:
    job = queue.find_one_and_update(
        {"status": "pending"},
        {"$set": {"status": "processing"}}
    )

    if not job:
        continue

    result = predict_caption(job["caption"])

    print("Caption:", job["caption"])
    print("Decision:", result["decision"])
    print("XGBoost Score:", result["score"])
    print("OOD:", result["ood"])

    print("-" * 40)

    final.insert_one({
    "caption": job["caption"],
    "imageId": job.get("imageId"),
    "decision": result["decision"],
    "score": result["score"],
    "ood": result["ood"],
})


    queue.delete_one({"_id": job["_id"]})
