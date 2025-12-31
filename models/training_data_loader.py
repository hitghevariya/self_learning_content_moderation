import joblib
import os

BASE_DIR = "models"
X_PATH = os.path.join(BASE_DIR, "X_train.pkl")
Y_PATH = os.path.join(BASE_DIR, "y_train.pkl")

def load_old_training_data():
    """
    Load historical training data for continual learning
    """
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        raise RuntimeError(
            " Training data not found. Run initial training first."
        )

    X_old = joblib.load(X_PATH)
    y_old = joblib.load(Y_PATH)

    return X_old, y_old
