import os
import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from config import WORD2VEC_MODEL_PATH
from models.training_xgboost import retrain_model
from models.feature_extraction import extract_features
from wordtovec.captions_to_vectors import captions_to_vectors
from datafetcher.open_data_fetcher import fetch_open_data
from database.novel_inputs import mark_novel_inputs_used,has_unused_novel_inputs


X_TRAIN_PATH = "models/X_train.pkl"
Y_TRAIN_PATH = "models/y_train.pkl"
W2V_MODEL = Word2Vec.load(WORD2VEC_MODEL_PATH)


FEATURE_COLS = [
    "token_count",
    "numeric_ratio",
    "vector_norm",
    "vector_mean",
    "vector_std",
    "vector_min",
    "vector_max",
    "is_zero_vector",
    "insult_density",
]


from models.ood_detector import is_ood

def build_features_from_open_data(df):
    X_new, y_new = [], []

    for _, row in df.iterrows():
        text = str(row["text"])
        label = int(row["label"])

        vec = captions_to_vectors(text, W2V_MODEL)
        features = extract_features(text, vec)

        feature_vector = [features[c] for c in FEATURE_COLS]

        # ✅ CORRECT NOVELTY CHECK
        if not is_ood(feature_vector):
            continue

        X_new.append(feature_vector)
        y_new.append(label)

    return np.array(X_new), np.array(y_new)


def run_open_data_training():
    print("Fetching open data...")
    df = fetch_open_data()

    if df is None or len(df) == 0:
        print(" No open data fetched")
        return

    print(f" Open samples fetched: {len(df)}")

    X_new, y_new = build_features_from_open_data(df)
    if X_new.shape[0] < 10:
        print("Not enough new samples. Skipping retraining.")
        return


    if len(X_new) == 0:
        print(" No usable training samples")
        return

    if not os.path.exists(X_TRAIN_PATH) or not os.path.exists(Y_TRAIN_PATH):
        print(" Training memory not found. Run base training first.")
        return

    X_old = joblib.load(X_TRAIN_PATH)
    y_old = joblib.load(Y_TRAIN_PATH)

    print(" Retraining XGBoost with open data...")
    improved = retrain_model(
        X_old=X_old,
        y_old=y_old,
        X_new=X_new,
        y_new=y_new
    )

    if improved:
        mark_novel_inputs_used()
        print(" Novel inputs marked as used")


if __name__ == "__main__":
    run_open_data_training()
