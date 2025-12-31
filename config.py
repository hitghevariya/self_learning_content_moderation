import os
from dotenv import load_dotenv

load_dotenv()

# ENV
ENV = os.getenv("ENV", "development")
PORT = os.getenv("PORT")


# MODEL PATHS
WORD2VEC_MODEL_PATH = os.getenv("WORD2VEC_MODEL_PATH")
XGBOOST_MODEL_PATH = os.getenv("XGBOOST_MODEL_PATH")
SCALER_PATH = os.getenv("SCALER_PATH")

# THRESHOLDS
ABUSIVE_THRESHOLD = float(os.getenv("ABUSIVE_THRESHOLD", 0.65))
BORDERLINE_THRESHOLD = float(os.getenv("BORDERLINE_THRESHOLD", 0.50))



# DB
MONGO = os.getenv("MONGO")
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_name")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

cloud_name = os.getenv("cloud_name")
api_key = os.getenv("apikey")
api_secret = os.getenv("api_secret")

Qdrant_url = os.getenv("Qdrant_url")
Qdrant_api_key = os.getenv("Qdrant_api")

XGBOOST_data = os.getenv("XGBOOST_DATA")

# LOGGING
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
