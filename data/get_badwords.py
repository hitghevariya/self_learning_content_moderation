from qdrant_client import QdrantClient
from config import Qdrant_api_key,Qdrant_url

client = QdrantClient(
    url=(Qdrant_url),
    api_key=(Qdrant_api_key),
)

def fetch_bad_words(limit=1000):
    """
    Fetch bad words stored in Qdrant collection `bad_words`
    """
    words = set()
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name="bad_words",
            limit=limit,
            offset=offset,
            with_payload=True
        )

        for p in points:
            if "word" in p.payload:
                words.add(p.payload["word"])

        if offset is None:
            break

    return list(words)

