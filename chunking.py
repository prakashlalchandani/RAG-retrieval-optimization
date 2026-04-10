from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

def create_chunks(pdf_path):
    # 1. Extract elements (same as before)
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True
    )

    # 2. SEMANTIC GROUPING (The Magic Fix)
    # This automatically merges orphaned values (like ": Rs. 200000/-") 
    # with their headers, creating highly contextual blocks of text.
    chunked_elements = chunk_by_title(
        elements,
        combine_text_under_n_chars=300, # Merges small fragments together
        max_characters=1500 # Prevents the chunk from getting too large for the LLM
    )

    chunks = []
    
    for element in chunked_elements:
        # Instead of trying to force HTML to Markdown which failed on your EMI table,
        # we let `chunk_by_title` provide the cleaned, grouped text representation.
        text = str(element).strip()
        
        # Optional: Clean up any weird OCR artifact spaces
        text = " ".join(text.split()) 
        
        if text:
            chunks.append(text)

    return chunks