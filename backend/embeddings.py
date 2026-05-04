import os
import time
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError

load_dotenv()

# The new SDK automatically detects the GEMINI_API_KEY in your environment
client = genai.Client()

def create_embeddings(chunks):
    """Generates high-dimensional embeddings using safe batches and rate limit handling."""
    
    all_embeddings = []
    
    # Google's API limit is 100 chunks per request
    batch_size = 100
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        
        # Keep trying to send this batch until it succeeds
        while True:
            try:
                response = client.models.embed_content(
                    model="gemini-embedding-001", 
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                )
                
                # Extract the vector arrays from the batch
                batch_embeddings = [e.values for e in response.embeddings]
                all_embeddings.extend(batch_embeddings)
                
                print(f"Successfully embedded batch {(i // batch_size) + 1} of {(len(chunks) // batch_size) + 1}...")
                
                # Success! Break the while loop to move on to the next batch
                break 
                
            except ClientError as e:
                # Check if the error is exactly the 429 Rate Limit
                if e.code == 429:
                    print("Google Free Tier Limit Hit! Pausing server for 60 seconds to reset quota...")
                    time.sleep(60) # Freeze the code for 1 minute
                else:
                    # If it is any other type of error, crash the program normally
                    raise e
    
    return np.array(all_embeddings)