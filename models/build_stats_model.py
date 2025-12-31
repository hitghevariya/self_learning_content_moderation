import joblib
import numpy as np
import os

X_TRAIN_PATH = "models/X_train.pkl"
STATS_PATH = "models/embedding_stats.pkl"


def build_stats_model():
    if not os.path.exists(X_TRAIN_PATH):
        raise RuntimeError(" X_train.pkl not found. Train baseline model first.")

    X = joblib.load(X_TRAIN_PATH)

    # Per-feature stats
    stats = {
        "mean": np.mean(X, axis=0),
        "std": np.std(X, axis=0) + 1e-8,  # avoid divide by zero
        "min": np.min(X, axis=0),
        "max": np.max(X, axis=0),

        # Overall vector norm stats
        "norm_mean": float(np.mean(np.linalg.norm(X, axis=1))),
        "norm_std": float(np.std(np.linalg.norm(X, axis=1)) + 1e-8),
    }

    joblib.dump(stats, STATS_PATH)
    print(" Stats model saved → models/embedding_stats.pkl")


if __name__ == "__main__":
    build_stats_model()
