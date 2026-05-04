from qdrant_client import QdrantClient

# 1. Local storage se connect karein
client = QdrantClient(path="./qdrant_storage") 

def get_latest_collection():
    """Database mein se sabse naya loan_agreements collection dhoondta hai"""
    collections = client.get_collections().collections
    # Sirf wo naam nikalein jo 'loan_agreements' se shuru hote hain
    names = [c.name for c in collections if c.name.startswith("loan_agreements")]
    # Sabse aakhri (latest) waala return karein
    return names[-1] if names else None

collection_name = get_latest_collection()

if not collection_name:
    print("Bhai, koi bhi collection nahi mila! Pehle document upload karein.")
else:
    try:
        print(f"✅ Found Active Collection: {collection_name}")
        print(f"Fetching data...\n")
        
        # 3. 'scroll' method se data nikalna
        results, _ = client.scroll(
            collection_name=collection_name,
            limit=3,             
            with_payload=True,   
            with_vectors=True    
        )

        if not results:
            print("Collection toh mil gaya par usme koi data (vectors) nahi hain.")

        for point in results:
            print(f"🔹 Point ID: {point.id}")
            print(f"🔹 Text Snippet: {point.payload.get('text', '')[:100]}...")
            if point.vector:
                print(f"🔹 Vector (First 5): {point.vector[:5]}...") 
            print("-" * 60)

    except Exception as e:
        print(f"Error aagaya bhai: {e}")