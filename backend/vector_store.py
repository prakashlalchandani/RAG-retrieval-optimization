from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

# Initialize Qdrant persistent storage
client = QdrantClient(path="./qdrant_storage")

def get_active_collection():
    """Finds the most recent collection after a server restart."""
    collections = client.get_collections().collections
    names = [c.name for c in collections if c.name.startswith("loan_agreements")]
    return names[-1] if names else "loan_agreements"

# Automatically locate the database name when the server boots up
current_collection_name = get_active_collection()


def build_qdrant_index(embeddings, chunks, filename):
    global current_collection_name
    dimension = len(embeddings[0])

    # 1. Housekeeping: Delete old collections so we don't waste hard drive space
    collections = client.get_collections().collections
    for c in collections:
        if c.name.startswith("loan_agreements"):
            client.delete_collection(c.name)

    # 2. Generate a fresh, unique collection name
    current_collection_name = f"loan_agreements_{uuid.uuid4().hex[:8]}"

    print(f"Current collection name: ", current_collection_name)
    
    # 3. Create the clean collection
    client.create_collection(
        collection_name=current_collection_name,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
    )

    # 4. Insert points
    points = []
    for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),
            payload={"text": chunk, "original_index": i, "document_name": filename} # <-- filename add kiya 
        ))

    client.upsert(
        collection_name=current_collection_name,
        points=points
    )
    return True


def search_qdrant(query_embedding, target_document=None, top_k=8):
    if not client.collection_exists(collection_name=current_collection_name):
        return []

    search_filter = None
    if target_document:
        search_filter = models.Filter(
            must=[models.FieldCondition(key="document_name", match=models.MatchValue(value=target_document))]
        )

    search_result = client.query_points(
        collection_name=current_collection_name,
        query=query_embedding.tolist(),
        query_filter=search_filter, # <-- Yahan filter lagaya
        limit=top_k
    ).points
    
    return [hit.payload["original_index"] for hit in search_result]


def get_all_chunks():
    """Extracts all text chunks directly from the Qdrant database payloads."""
    if not client.collection_exists(collection_name=current_collection_name):
        return []
        
    # Qdrant's scroll API fetches records without needing a search query
    records, _ = client.scroll(
        collection_name=current_collection_name,
        limit=10000,
        with_payload=True,
        with_vectors=False
    )
    
    # Sort by the original index so the document reads in the correct order
    sorted_records = sorted(records, key=lambda x: x.payload["original_index"])
    
    # Extract just the text strings
    return [record.payload["text"] for record in sorted_records]