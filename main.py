from chunking import extract_text, create_chunks
from embeddings import create_embeddings, model
from vector_store import build_faiss_index, search


text = extract_text("D:\\Projects\\internship\\rag-retrieval-optimization\\sample data\\loan_agreement.pdf")

chunks = create_chunks(text)

embeddings = create_embeddings(chunks)

index = build_faiss_index(embeddings)


query = "interest rate"

query_embedding = model.encode([query])

results = search(index, query_embedding)


print("\nRetrieved chunks:\n")

for idx in results[0]:
    print(chunks[idx])
    print("\n------------------\n")