import os
import json
from dotenv import load_dotenv
from groq import Groq
from tools import calculate, calculator_tool

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def generate_answer(query, retrieved_chunks):
    # 1. Provide the retrieved text to the LLM
    context = "\n\n".join(retrieved_chunks)
    
    # 2. Update the System Prompt to enforce Agentic behavior
    system_prompt = (
        "You are an elite autonomous financial auditor. You analyze retrieved legal and financial documents. "
        "If a user asks a question that requires calculating totals, differences, or multiplications (like total interest paid), "
        "you MUST extract the raw numbers from the context and use your 'calculate' tool to do the math. "
        "Never guess mathematical answers."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    print("\n[Agent] Analyzing request...")
    
    # 3. The First Pass: Groq decides whether to answer or use a tool
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        tools=[calculator_tool], # Pass the tool schema from tools.py
        tool_choice="auto"       # Let the AI decide when it needs the tool
    )

    response_message = response.choices[0].message

    # 4. The Agentic Loop: Did Groq call a tool?
    if response_message.tool_calls:
        # Save Groq's tool request to the message history
        messages.append(response_message)
        
        for tool_call in response_message.tool_calls:
            if tool_call.function.name == "calculate":
                # Extract the equation Groq built
                args = json.loads(tool_call.function.arguments)
                equation = args.get("equation")
                
                # --- THIS IS THE MAGIC: Your Python Backend executing the LLM's request ---
                print(f"[Agent Tool Execution] Groq requested math: {equation}")
                tool_result = calculate(equation)
                print(f"[Agent Tool Execution] Backend calculated: {tool_result}")
                # --------------------------------------------------------------------------
                
                # Give the hard mathematical truth back to Groq
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": tool_result
                })

        # 5. The Second Pass: Groq writes the final human response using the tool's output
        print("[Agent] Synthesizing final response...")
        final_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages
        )
        return final_response.choices[0].message.content

    # If no math was needed, just return the standard text answer
    return response_message.content