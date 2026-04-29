import os
from dotenv import load_dotenv
from groq import Groq
from unstructured.partition.pdf import partition_pdf
from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser

# Load environment variables to securely access the GROQ_API_KEY
load_dotenv()

# Initialize the Groq client to summarize tables during upload
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def summarize_table(table_html):
    """Uses the lightning-fast Groq model to convert a raw table into natural sentences."""
    prompt = f"""
    You are an expert data analyst. Read the following HTML table extracted from a legal PDF. 
    Write a highly detailed, natural language summary of every key-value pair and data point in this table.
    State the numbers clearly. Do not use formatting, just plain sentences.
    
    Table Data:
    {table_html}
    """
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Warning: Table summarization failed. Error: {e}")
        return "Table summarization unavailable."

def create_chunks(pdf_path):
    print("Extracting text and tables using Unstructured...")
    
    # 1. The Pro-Level Table Extractor (Unstructured)
    # This reads the PDF using advanced layout-awareness to keep tables perfectly intact.
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True
    )
    
    processed_texts = []
    
    # 2. Intercept Tables for LLM Summarization
    for el in elements:
        # If the element is a Table, convert it to a natural language summary!
        if el.category == "Table":
            print("Table detected! Generating semantic summary with Groq...")
            # Unstructured stores the raw HTML of the table in metadata
            table_html = el.metadata.text_as_html if hasattr(el.metadata, 'text_as_html') and el.metadata.text_as_html else str(el)
            
            # Ask Groq to write a paragraph about this table
            table_summary = summarize_table(table_html)
            
            # Append BOTH the raw table (for BM25 exact match) AND the summary (for Vector match)
            processed_texts.append(f"RAW TABLE DATA:\n{str(el)}\n\nTABLE SUMMARY:\n{table_summary}")
        else:
            # If it is normal text, just keep it as is
            processed_texts.append(str(el).strip())
    
    clean_text = "\n\n".join([text for text in processed_texts if text])
    
    # 3. The Bridge: Wrap the clean text into a LlamaIndex Document object
    document = Document(text=clean_text)
    
    # 4. The Enterprise Chunking Tree (LlamaIndex)
    print("Building Hierarchical nodes...")
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[2048, 512, 128]
    )
    nodes = node_parser.get_nodes_from_documents([document])
    
    # 5. Extract the final text chunks for Qdrant and BM25
    chunks = []
    for node in nodes:
        text = node.text.strip()
        if text:
            # Clean up excessive spacing
            text = " ".join(text.split()) 
            chunks.append(text)

    return chunks