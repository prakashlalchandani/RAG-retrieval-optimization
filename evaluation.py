import re

test_questions = {

    "What is EMI amount?": "32,425",

    "What is interest rate?": "10.75",

    "Loan amount sanctioned?": "1,500,000",

    "Number of installments?": "60"
}

def normalize(text):
    return re.sub(r"[^\w]", "", text.lower())

def check_retrieval(expected_answer, retrieved_chunks):

    expected_clean = normalize(expected_answer)

    for chunk in retrieved_chunks:

        chunk_clean = normalize(chunk)

        if expected_clean in chunk_clean:
            return True

    return False
