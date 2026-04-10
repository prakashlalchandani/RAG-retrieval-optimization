from chunking import create_chunks
from embeddings import create_embeddings, model
from vector_store import build_faiss_index, search
from hybrid_search import build_bm25, bm25_search, reciprocal_rank_fusion
from evaluation import test_questions, check_retrieval
from query_transforms import HYDE_CONFIG, generate_query_variants, text_for_retrieval


def build_query_set(query: str, variant_count: int = 4):
    query_set = []
    for candidate in [query] + generate_query_variants(query, n=variant_count):
        normalized = " ".join(candidate.strip().split())
        if normalized and normalized not in query_set:
            query_set.append(normalized)
    return query_set


def run_evaluation(pdf_path):

    chunks = create_chunks(pdf_path)

    embeddings = create_embeddings(chunks)

    index = build_faiss_index(embeddings)

    bm25 = build_bm25(chunks)

    correct = 0
    total = len(test_questions)

    for query, expected_answer in test_questions.items():
        rank_lists = []
        query_set = build_query_set(query.lower())
        used_hyde = False
        hyde_error = None

        for query_variant in query_set:
            retrieval_text, variant_used_hyde, variant_hyde_error = text_for_retrieval(query_variant)
            query_embedding = model.encode([retrieval_text])

            vector_results = list(search(index, query_embedding, top_k=8)[0])
            bm25_results = bm25_search(bm25, query_variant, chunks, top_k=8)

            rank_lists.extend([vector_results, bm25_results])
            used_hyde = used_hyde or variant_used_hyde
            if variant_hyde_error and hyde_error is None:
                hyde_error = variant_hyde_error

        final_results = reciprocal_rank_fusion(rank_lists)[:5]

        retrieved_chunks = [chunks[i] for i in final_results]

        result = check_retrieval(expected_answer, retrieved_chunks)

        print("\nQuery:", query)
        print("HyDE Enabled:", HYDE_CONFIG.enabled)
        print("HyDE Used:", used_hyde)
        if hyde_error:
            print("HyDE Fallback Reason:", hyde_error)
        print("Match Found:", result)

        if result:
            correct += 1

    accuracy = correct / total

    print("\nFinal Accuracy:", accuracy)


if __name__ == "__main__":

    pdf_path = input("Enter PDF path for evaluation: ")

    run_evaluation(pdf_path)
