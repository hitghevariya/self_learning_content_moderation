from gensim.models import Word2Vec
import numpy as np

W2V_PATH = "wordtovec/custom_word2vec.model"

def load_bad_word_vectors(bad_words):
    model = Word2Vec.load(W2V_PATH)
    vectors = []

    for word in bad_words:
        if word in model.wv:
            vectors.append(model.wv[word])

    return np.array(vectors)
