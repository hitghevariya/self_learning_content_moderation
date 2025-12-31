from qdrant_client import QdrantClient
from config import Qdrant_url,Qdrant_api_key


client = QdrantClient(url=Qdrant_url, 
    api_key=Qdrant_api_key,)

def check_badword_similarity(vector, threshold=0.25):
    results = client.search(
        collection_name="bad_words",
        query_vector=vector.tolist(),
        limit=5
    )

    matches = []

    for res in results:
        if res.score >= threshold:
            matches.append({
                "word": res.payload.get("word"),
                "score": res.score
            })

    if matches:
        return True, matches
    else:
        return False, []

