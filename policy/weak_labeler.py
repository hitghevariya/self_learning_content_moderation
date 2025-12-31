from sklearn.metrics.pairwise import cosine_similarity

def auto_label(vec, bad_vecs):
    score = cosine_similarity([vec], bad_vecs).max()
    if score > 0.75:
        return 1
    if score < 0.35:
        return 0
    return None
