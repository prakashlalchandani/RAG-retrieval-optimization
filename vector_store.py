from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

# 1. Initialize Qdrant in persistent local mode. 
# This creates a folder called "qdrant_storage" in your project directory.
client = QdrantClient(path="./qdrant_storage")

# We name our database collection
COLLECTION_NAME = "loan_agreements"

def init_qdrant(dimension: int):
    """Creates the collection if it doesn't already exist."""
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )

def build_qdrant_index(embeddings, chunks):
    """Takes embeddings and chunks and saves them to the persistent database."""
    dimension = len(embeddings[0])
    init_qdrant(dimension)

    points = []
    for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
        # Qdrant requires a unique ID for every single vector
        point_id = str(uuid.uuid4())
        
        points.append(PointStruct(
            id=point_id,
            vector=embedding.tolist(), # Convert numpy array to standard list
            # We save the chunk's text and original index directly inside the database!
            payload={"text": chunk, "original_index": i} 
        ))

    # Push to database
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    return True

def search_qdrant(query_embedding, top_k=8):
    """Searches the persistent database and returns the original chunk indices."""
    
    # Check if we even have a database yet
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        return []

    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding.tolist(),
        limit=top_k
    )
    
    # To keep this perfectly compatible with your existing hybrid_search logic,
    # we return a list of the original indices (e.g., [14, 2, 7])
    indices = [hit.payload["original_index"] for hit in search_result]
    return indices