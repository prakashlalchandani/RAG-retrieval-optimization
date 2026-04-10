from chunking import create_chunks
from embeddings import create_embeddings, model
from vector_store import build_faiss_index, search
from hybrid_search import build_bm25, bm25_search, hybrid_search
from evaluation import test_questions, check_retrieval
from query_transforms import HYDE_CONFIG, text_for_retrieval


def run_evaluation(pdf_path):

    chunks = create_chunks(pdf_path)

    embeddings = create_embeddings(chunks)

    index = build_faiss_index(embeddings)

    bm25 = build_bm25(chunks)

    correct = 0
    total = len(test_questions)

    for query, expected_answer in test_questions.items():

        retrieval_text, used_hyde, hyde_error = text_for_retrieval(query.lower())
        query_embedding = model.encode([retrieval_text])

        vector_results = list(search(index, query_embedding)[0])

        bm25_results = bm25_search(bm25, query, chunks)

        final_results = hybrid_search(vector_results, bm25_results)

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