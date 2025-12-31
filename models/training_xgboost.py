import pandas as pd
import joblib
import numpy as np
import xgboost as xgb
import hashlib
import os


from config import XGBOOST_MODEL_PATH, XGBOOST_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

MODEL_PATH = XGBOOST_MODEL_PATH
CSV_PATH = XGBOOST_data
X_TRAIN_PATH = "models/X_train.pkl"
Y_TRAIN_PATH = "models/y_train.pkl"
def _hash_dataset(X, y):
    """
    Create a deterministic hash for training data
    """
    h = hashlib.md5()
    h.update(X.tobytes())
    h.update(y.tobytes())
    return h.hexdigest()



def load_model():
    """
    Load existing XGBoost model if present
    """
    if MODEL_PATH is None:
        return None

    if not os.path.exists(MODEL_PATH):
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"⚠️ Failed to load old model: {e}")
        return None


def build_training_dataset():
    if CSV_PATH is None:
        raise RuntimeError("❌ CSV_PATH is None")

    if not os.path.exists(CSV_PATH):
        raise RuntimeError(f"❌ CSV file not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print("📄 CSV columns:", df.columns.tolist())

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

    LABEL_COL = "label"

    X = df[FEATURE_COLS].values.astype(float)
    y = df[LABEL_COL].values.astype(int)

    print(f"✅ Loaded {len(X)} samples with {X.shape[1]} features")
    return X, y

def retrain_model(X_old, y_old, X_new, y_new):

    # Safe merge
    X = X_old if len(X_new) == 0 else np.vstack([X_old, X_new])
    y = y_old if len(y_new) == 0 else np.concatenate([y_old, y_new])
    DATA_HASH_FILE = "models/data_hash.txt"

    new_hash = _hash_dataset(X, y)

    
    if not os.path.exists(DATA_HASH_FILE):
        print(" No existing data hash found. Forcing initial training.")
    else:
        with open(DATA_HASH_FILE, "r") as f:
            old_hash = f.read().strip()

        if new_hash == old_hash:
            print(" Training data unchanged. Skipping retraining.")
            return False


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    old_model = load_model()
    old_score = 0.0

    if old_model is not None:
        old_score = f1_score(y_test, old_model.predict(X_test))

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    new_score = f1_score(y_test, model.predict(X_test))

    print(f"Old F1: {old_score:.4f}")
    print(f"New F1: {new_score:.4f}")

   # 🔑 BASELINE SAVE: if training memory does not exist
    if not os.path.exists(X_TRAIN_PATH) or not os.path.exists(Y_TRAIN_PATH):
        joblib.dump(model, MODEL_PATH)
        joblib.dump(X, X_TRAIN_PATH)
        joblib.dump(y, Y_TRAIN_PATH)
        print("✅ Baseline training data saved")
        return True

    # 🔁 NORMAL CONTINUAL LEARNING
    if new_score == old_score:
        joblib.dump(model, MODEL_PATH)
        joblib.dump(X, X_TRAIN_PATH)
        joblib.dump(y, Y_TRAIN_PATH)
        with open(DATA_HASH_FILE, "w") as f:
            f.write(new_hash)
        print("✅ Improved model deployed")
        return True

    print("❌ Model rejected")
    return False


if __name__ == "__main__":
    print("🚀 Starting initial XGBoost training from CSV")

    X, y = build_training_dataset()

    retrain_model(
        X_old=X,
        y_old=y,
        X_new=np.empty((0, X.shape[1])),
        y_new=np.array([])
    )

    print("Initial training completed")
