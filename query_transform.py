import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

load_dotenv()
client = genai.Client()

def generate_hyde_document(query: str, max_retries=3) -> str:
    prompt = f"""
    You are a financial legal expert. A user has asked this question: '{query}'
    Write a short, formal 2-sentence excerpt from a loan agreement that would answer this question.
    Do not use introductory filler. Just write the formal legal text.
    """
    
    # Retry Loop
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return response.text
            
        except APIError as e: # Catch the Google API error
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt # Waits 1s, then 2s, then 4s
                print(f"API busy. Retrying HyDE in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"API completely unavailable. Skipping HyDE: {e}")
                return "" # Return empty string so the app doesn't crash

def generate_multi_queries(query: str, max_retries=3) -> list:
    prompt = f"""
    Rewrite the following search query into 3 different variations using formal financial and banking terminology.
    Return only the queries, separated by newlines. Do not use bullet points or numbers.
    Original query: '{query}'
    """
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return [q.strip() for q in response.text.strip().split('\n') if q.strip()]
            
        except APIError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"API busy. Retrying Multi-Query in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"API completely unavailable. Skipping Multi-Query: {e}")
                return [] # Return empty list so the app doesn't crash