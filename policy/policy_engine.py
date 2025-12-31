import re
import numpy as np

from models.load_model import load_embedding_model
from wordtovec.captions_to_vectors import captions_to_vectors
from database.qdrant_check import check_badword_similarity
from models.feature_extraction import extract_features
import joblib
from config import XGBOOST_MODEL_PATH
# Load models ONCE
xgb_model = joblib.load(XGBOOST_MODEL_PATH)
embedding_model = load_embedding_model()


def evaluate_caption(caption: str):
    caption_clean = caption.strip().lower()

    if re.fullmatch(r"\d+", caption_clean):
        return decision(caption, "CLEAN", 0.0, [])

    if len(caption_clean) < 2:
        return decision(caption, "CLEAN", 0.0, [])

   
    vector = captions_to_vectors([caption], embedding_model)[0]

    if np.all(vector == 0):
        matches = []
    else:
        _, matches = check_badword_similarity(vector)

    features = extract_features(caption, matches, vector)
    X = np.array([features])

    
    #  XGBOOST PROBABILITY
   
    prob = float(xgb_model.predict_proba(X)[0][1])

    # POLICY DECISION
   
    # High confidence XGBoost
    if prob >= 0.90:
        return decision(caption, "ABUSIVE", prob, matches, prob)

    # Low confidence
    if prob < 0.30:
        return decision(caption, "CLEAN", prob, matches, prob)

# Borderline → vector similarity
    max_sim = max([m["score"] for m in matches], default=0.0)

    if max_sim >= 0.45:
        return decision(caption, "ABUSIVE", prob, matches, max_sim)

    return decision(caption, "CLEAN", prob, matches, max_sim)

def decision(caption, label, xgb_score, matches, final_score):
    return {
        "caption": caption,
        "decision": label,
        "xgboost_score": round(xgb_score, 3),
        "final_score": round(final_score, 3),
        "matched_words": [m["word"] for m in matches]
    }
