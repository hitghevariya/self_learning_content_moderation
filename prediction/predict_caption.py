# import numpy as np
# import joblib
# import re

# from load_model import load_embedding_model
# from captions_to_vectors import captions_to_vectors
# from qdrant_check import check_badword_similarity
# from feature_extraction import extract_features
# from mongo_fetch import fetch_captions_from_mongo


# # Load models
# xgb_model = joblib.load("xgboost_abuse_model2.pkl")
# embedding_model = load_embedding_model()
# print("Loaded model type:", type(xgb_model))




# def predict_caption(caption: str):
#     #if only num
#     if re.fullmatch(r"\d+", caption.strip()):
#         return {
#             "caption": caption,
#             "prediction": "CLEAN",
#             "confidence": 0.0,
#             "matched_words": []
#         }

#     # Rule 2: too short
#     if len(caption.strip()) <= 2:
#         return {
#             "caption": caption,
#             "prediction": "CLEAN",
#             "confidence": 0.0,
#             "matched_words": []
#         }
#     # Convert caption → vector
#     vector = captions_to_vectors([caption], embedding_model)[0]

#     # Qdrant similarity check
#     if np.all(vector == 0):
#         matches = []
#         is_abusive_rule = False
#     else:
#         is_abusive_rule, matches = check_badword_similarity(vector)

#     # Feature extraction
#     features = extract_features(caption, matches, vector)
#     X = np.array([features])


#     # XGBoost prediction
#     pred = xgb_model.predict(X)[0]
#     prob = xgb_model.predict_proba(X)[0][1]

#     return {
#         "caption": caption,
#         "prediction": "ABUSIVE" if pred == 1 else "CLEAN",
#         "confidence": round(prob, 3),
#         "matched_words": [m["word"] for m in matches],
#     }


# if __name__ == "__main__":
#     captions = fetch_captions_from_mongo()
    
#     print("Total captions from MongoDB:", len(captions))

#     for caption in captions:
#         result = predict_caption(caption)
#         print(result)
#         print("-" * 40)

import numpy as np
from config import (
    ABUSIVE_THRESHOLD,
    BORDERLINE_THRESHOLD,
    XGBOOST_MODEL_PATH,
    SCALER_PATH
)

import joblib
from preprocessing.text_cleaning import clean_caption
from wordtovec.captions_to_vectors import captions_to_vectors
from models.feature_extraction import extract_features
from models.load_model import load_embedding_model
from preprocessing.text_cleaning import clean_caption
import re
from models.ood_detector import is_ood
from database.novel_inputs import store_novel_input


def split_caption(caption: str):
    caption = caption.replace("#", " ")
    parts = re.split(r"\s+", caption)
    return [p for p in parts if len(p) > 2]


xgb_model = joblib.load(XGBOOST_MODEL_PATH)
embedding_model = load_embedding_model()

# def predict_caption(caption: str):
#     caption_clean = clean_caption(caption)
#     vector = captions_to_vectors([caption_clean], embedding_model)[0]

#     features = extract_features(caption_clean, vector)
#     X = np.array([list(features.values())])

#     prob = xgb_model.predict_proba(X)[0][1]

#     return {
#         "caption": caption,
#         "decision": "ABUSIVE" if prob >= 0.75 else "CLEAN",
#         "finalScore": round(float(prob), 3),
#         "xgboostScore": round(float(prob), 3)
#     }
POSITIVE_TOKENS = {
     # General positivity
    "good", "great", "awesome", "amazing", "excellent", "nice", "cool",
    "best", "beautiful", "wonderful", "fantastic", "perfect",

    # Social / friendly tone
    "friend", "bro", "buddy", "pal", "mate", "homie",
    "fam", "gang", "team",

    # Emotional positivity
    "happy", "joy", "fun", "smile", "laugh", "enjoy",
    "love", "peace", "calm", "relax", "chill", "vibe", "vibes",

    # Motivation / appreciation
    "proud", "respect", "support", "believe", "strong",
    "win", "winner", "success", "victory", "hustle",

    # Internet slang
    "lit", "dope", "fire", "slay", "goat"
}

POSITIVE_PROFANITY_PHRASES = {
    # Fuck-based (most common false positives)
    "fucking awesome",
    "fucking amazing",
    "fucking great",
    "fucking brilliant",
    "fucking cool",

    # Damn-based
    "damn good",
    "damn awesome",
    "damn amazing",

    # Bloody-based
    "bloody good",
    "bloody brilliant",
    "bloody awesome",

    # Slang appreciation
    "cool as hell",
    "awesome as hell"
}

def is_positive_context(caption: str) -> bool:
    text = caption.lower()

    for phrase in POSITIVE_PROFANITY_PHRASES:
        if phrase in text:
            return True

    positive_hits = sum(1 for w in POSITIVE_TOKENS if w in text)

    has_explicit_insult = any(
        bad in text for bad in {"idiot", "asshole", "stupid", "moron", "trash"}
    )

    return positive_hits >= 4 and not has_explicit_insult

def predict_caption(caption: str):
    """
    Full prediction pipeline:
    caption -> embedding -> features -> xgboost -> ood -> store
    """

    #  Convert caption → embedding
    vector = captions_to_vectors(caption, embedding_model)

    #  Extract numeric features (9 features)
    features = extract_features(caption, vector)

    feature_vector = [
        features["token_count"],
        features["numeric_ratio"],
        features["vector_norm"],
        features["vector_mean"],
        features["vector_std"],
        features["vector_min"],
        features["vector_max"],
        features["is_zero_vector"],
        features["insult_density"],
    ]

    x = np.array(feature_vector, dtype=float).reshape(1, -1)

    #  XGBoost score
    score = float(xgb_model.predict_proba(x)[0][1])

    # OOD detection (FEATURE SPACE ONLY)
    ood = is_ood(feature_vector)

    #  Decision thresholds
    if score >= ABUSIVE_THRESHOLD:
        decision = "ABUSIVE"
    elif score >= BORDERLINE_THRESHOLD:
        decision = "BORDERLINE"
    else:
        decision = "CLEAN"

    # OOD samples
    if is_ood(list(features.values())):
        store_novel_input(
            features=features,
            decision=decision,
            score=float(score)
        )

    
    return {
        "caption": caption,
        "decision": decision,
        "score": round(score, 4),
        "ood": ood,
        "features": features,
    }



def xgboost_similarity_score(text: str):
    caption_clean = clean_caption(text)
    vector = captions_to_vectors([caption_clean], embedding_model)[0]

    features = extract_features(caption_clean, vector)
    X = np.array([list(features.values())])

    prob = xgb_model.predict_proba(X)[0][1]
    return float(prob)

def average_xgboost_similarity(caption: str):
    parts = split_caption(caption)

    if not parts:
        return 0.0

    scores = []
    for part in parts:
        score = xgboost_similarity_score(part)

        # boost if multiple insults present
        tokens = part.lower().split()
        insult_count = sum(
            t in {"idiot", "asshole", "stupid", "moron", "trash", "loser"}
            for t in tokens
        )

        if insult_count >= 2:
            score *= 1.25  

        scores.append(min(score, 1.0))

    # Preserve strongest signal
    return round(
        max(scores) * 0.6 + (sum(scores) / len(scores)) * 0.4,
        3
    )


