import os
import time
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize the Groq client
# It will automatically detect os.environ["GROQ_API_KEY"]
client = Groq()

def generate_hyde_document(query: str, max_retries=3) -> str:
    """Generates a hypothetical financial document snippet using Groq."""
    prompt = f"""
    You are a financial legal expert. A user has asked this question: '{query}'
    Write a short, formal 2-sentence excerpt from a loan agreement that would answer this question.
    Do not use introductory filler. Just write the formal legal text.
    """
    
    # Retry Loop
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant", # Using the blazingly fast 8B model
                temperature=0.2 # Lower temperature keeps the text formal and factual
            )
            # Extract the text from the Groq response structure
            return response.choices[0].message.content.strip()
            
        except Exception as e: # Catch API errors
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt 
                print(f"Groq API busy. Retrying HyDE in {wait_time} seconds... Error: {e}")
                time.sleep(wait_time)
            else:
                print(f"Groq API completely unavailable. Skipping HyDE. Error: {e}")
                return ""

def generate_multi_queries(query: str, max_retries=3) -> list:
    """Rewrites the query into multiple formal variations using Groq."""
    prompt = f"""
    Rewrite the following search query into 3 different variations using formal financial and banking terminology.
    Return only the queries, separated by newlines. Do not use bullet points or numbers.
    Original query: '{query}'
    """
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.2
            )
            
            # Extract the text
            text_output = response.choices[0].message.content.strip()
            
            # Split the response into a list of strings
            return [q.strip() for q in text_output.split('\n') if q.strip()]
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Groq API busy. Retrying Multi-Query in {wait_time} seconds... Error: {e}")
                time.sleep(wait_time)
            else:
                print(f"Groq API completely unavailable. Skipping Multi-Query. Error: {e}")
                return []