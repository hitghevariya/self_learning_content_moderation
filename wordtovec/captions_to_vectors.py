import numpy as np
import re

def captions_to_vectors(captions, model):
    vectors = []
    vocab = list(model.wv.key_to_index.keys())

    for caption in captions:
        caption = caption.lower()

        # extract alphabetic tokens
        tokens = re.findall(r"[a-z]+", caption)

        word_vectors = []

        for token in tokens:
            for bad_word in vocab:
                #  BIDIRECTIONAL PARTIAL MATCH
                if bad_word in token or token in bad_word:
                    word_vectors.append(model.wv[bad_word])

        if word_vectors:
            sentence_vector = np.mean(word_vectors, axis=0)
        else:
            sentence_vector = np.zeros(model.vector_size)

        vectors.append(sentence_vector)

    return vectors
