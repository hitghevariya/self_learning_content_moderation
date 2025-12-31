import joblib
import numpy as np

STATS_PATH = "models/embedding_stats.pkl"

def is_ood(feature_vector, z_threshold=3.0):
    """
    feature_vector: list or np.array of length 9
    """

    stats = joblib.load(STATS_PATH)

    x = np.array(feature_vector, dtype=float)

    # Per-feature z-score check
    z_scores = np.abs((x - stats["mean"]) / stats["std"])
    if np.any(z_scores > z_threshold):
        return True

    # Overall norm check (feature space)
    norm = np.linalg.norm(x)
    if abs(norm - stats["norm_mean"]) > z_threshold * stats["norm_std"]:
        return True

    return False
