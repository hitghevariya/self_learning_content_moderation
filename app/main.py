# from pymongo import MongoClient
# from mongo_fetch import fetch_captions_from_mongo
# from load_model import load_embedding_model
# from captions_to_vectors import captions_to_vectors
# from qdrant_check import check_badword_similarity
# import numpy as np
# import re

# # prediction= "ABUSIVE"


# model = load_embedding_model()

# # Fetch captions from MongoDB
# captions = fetch_captions_from_mongo()
# # if re.fullmatch(r"\d+", captions.strip()):
# #         return {
# #             "caption": captions,
# #             "prediction": "CLEAN",
# #             "confidence": 0.0,
# #             "matched_words": []
# #         }

# #     # Rule 2: too short
# # if len(captions.strip()) <= 2:
# #         return {
# #             "caption": captions,
# #             "prediction": "CLEAN",
# #             "confidence": 0.0,
# #             "matched_words": []
# #         }


# # Convert captions to vectors
# vectors = captions_to_vectors(captions, model)


# print("Captions:", captions)
# print("Vector count:", len(vectors))
# print("Vector size:", len(vectors[0]))

# #  Check against Qdrant bad-word vectors
# for caption, vector in zip(captions, vectors):
#     is_abusive, match = check_badword_similarity(vector)
#     if np.all(vector == 0):
#         print("Caption:", caption)
#         print("No known bad words → CLEAN (skipped)")
#         print("-" * 40)
#         continue
#     if np.all(vector == 0) is False:
#         is_abusive, match = check_badword_similarity(vector, threshold=0.25)
#     is_abusive, matches = check_badword_similarity(vector)

#     print("Caption:", caption)

#     if is_abusive:
#         print("ABUSIVE")
#         for m in matches:
#             print("Matched word:", m["word"], "Score:", m["score"])
#     else:
#         print("CLEAN")

# print("-" * 40)

from database.mongo_fetch import fetch_captions_from_mongo
from policy.policy_engine import evaluate_caption


def main():
    print(" Starting moderation pipeline...\n")

    captions = fetch_captions_from_mongo()
    print(f" Total captions fetched: {len(captions)}\n")

    for caption in captions:
        result = evaluate_caption(caption)

        print("Caption:", result["caption"])
        print("Final Decision:", result["decision"])
        print("XGBoost Score:", result["xgboost_score"])
        print("Matched Words:", result["matched_words"])
        print("-" * 50)


if __name__ == "__main__":
    main()
