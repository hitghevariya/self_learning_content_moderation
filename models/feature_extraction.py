# import re
# import numpy as np



# FEATURE_COLUMNS = [
#     "max_similarity",
#     "mean_similarity",
#     "bad_word_count",
#     "unique_bad_words",
#     "token_count",
#     "hashtag_count",
#     "contains_number",
#     "zero_vector",
# ]

# def extract_features(caption, matches, vector):
    
#     tokens = re.findall(r"[a-z]+", caption.lower())

#     feature_dict = {
#         "max_similarity": max([m["score"] for m in matches], default=0.0),
#         "mean_similarity": (
#             sum(m["score"] for m in matches) / len(matches)
#             if matches else 0.0
#         ),
#         "bad_word_count": len(matches),
#         "unique_bad_words": len(set(m["word"] for m in matches)),
#         "token_count": len(tokens),
#         "hashtag_count": caption.count("#"),
#         "contains_number": int(any(c.isdigit() for c in caption)),
#         "zero_vector": int(np.all(vector == 0)),
#     }

#     # RETURN FEATURES IN FIXED ORDER
#     return [feature_dict[col] for col in FEATURE_COLUMNS]
import numpy as np

def extract_features(caption, vector):
    tokens = caption.split()
    insult_density = sum(1 for t in tokens if t in {"idiot", "asshole", "stupid", "moron", "trash"}
    ) / max(len(tokens), 1)
    is_zero_vector = int(np.all(vector == 0))

    return {
        "token_count": len(tokens),
        "numeric_ratio": sum(t.isdigit() for t in tokens) / max(len(tokens), 1),
        "vector_norm": float(np.linalg.norm(vector)),
        "vector_mean": float(np.mean(vector)),
        "vector_std": float(np.std(vector)),
        "vector_min": float(np.min(vector)),
        "vector_max": float(np.max(vector)),
        "is_zero_vector": is_zero_vector ,
        "insult_density": insult_density,
    }
