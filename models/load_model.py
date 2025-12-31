from gensim.models import Word2Vec
from config import WORD2VEC_MODEL_PATH

def load_embedding_model():
    model = Word2Vec.load(WORD2VEC_MODEL_PATH)
    return model
