from database.novel_inputs import novel_inputs
from policy.weak_labeler import weak_label
from data.get_badwords import fetch_bad_words

from .training_data_loader import load_old_training_data
from .training_xgboost import retrain_model
from .vector_loaders import load_bad_word_vectors


def run_autonomous_training():
    samples = list(novel_inputs.find({"used": False}))

    if len(samples) < 50:
        print("Not enough novel samples")
        return

    X_old, y_old = load_old_training_data()

    bad_words = fetch_bad_words()
    bad_vectors = load_bad_word_vectors(bad_words)

    X_new, y_new = [], []

    for s in samples:
        label = weak_label(s["vector"], bad_vectors)
        if label is not None:
            X_new.append(s["vector"])
            y_new.append(label)

    if len(X_new) < 20:
        print("Low-confidence data, skipping retrain")
        return

    retrain_model(X_old, y_old, X_new, y_new)

    novel_inputs.update_many({"used": False}, {"$set": {"used": True}})
