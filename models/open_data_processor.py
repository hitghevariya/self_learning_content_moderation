import pandas as pd
import numpy as np
from gensim.models import Word2Vec

from config import WORD2VEC_MODEL_PATH
from models.feature_extraction import extract_features
from wordtovec import captions_to_vectors

def process_open_data(csv_path):
    model = Word2Vec.load(WORD2VEC_MODEL_PATH)
    df = pd.read_csv(csv_path)

    X, y = [], []

    for _, row in df.iterrows():
        text = str(row["text"])
        label = int(row["label"])

        vec = captions_to_vectors(text, model)
        features = extract_features(text, vec)

        X.append(list(features.values()))
        y.append(label)

    return np.array(X), np.array(y)
