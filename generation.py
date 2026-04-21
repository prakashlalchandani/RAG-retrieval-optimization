import os
from google import genai

client = genai.Client()

def generate_final_answer(query: str, retrieved_chunks: list) -> str:
    # Combine the retrieved chunks into one context block
    context = "\n\n".join(retrieved_chunks)
    
    prompt = f"""
    You are a highly precise financial extraction system.
    Your task is to answer the user's query using ONLY the provided context.
    
    CRITICAL RULES:
    1. Be as concise as humanly possible. 
    2. If the answer is a single phrase (like "Personal Loan" or "13.65%"), return ONLY that phrase.
    3. Do not use conversational filler like "The bank will..." or "According to the document...".
    4. If the answer is not in the context, output exactly: "Information not found in document."
    
    Query: {query}
    
    Context:
    {context}
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    return response.text.strip()