import numpy as np
import pandas as pd

from models.load_model import load_embedding_model
from wordtovec.captions_to_vectors import captions_to_vectors
from models.feature_extraction import extract_features
from data.synthetic_data import generate_dataset

# Load embedding model (UNCHANGED)
embedding_model = load_embedding_model()

# Generate new synthetic data
data = generate_dataset(size=100000)

captions = [c for c, _ in data]
labels = [l for _, l in data]

# Convert captions → vectors
vectors = captions_to_vectors(captions, embedding_model)

X = []
y = []

for caption, vector, label in zip(captions, vectors, labels):
    features = extract_features(caption, vector)
    X.append(features)
    y.append(label)

# Save dataset
df = pd.DataFrame(X)
df["label"] = y

df.to_csv("synthetic_abuse_dataset2.csv", index=False)
print("Dataset generated:", df.shape)
