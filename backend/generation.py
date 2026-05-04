import os
import json
from dotenv import load_dotenv
from groq import Groq
from tools import calculate, calculator_tool

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def generate_answer(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    
    # --- THE UPGRADE: Strict Chain-of-Thought Prompting ---
    # --- THE ULTIMATE SYNTAX FIX ---
    # --- THE IRON-CLAD GUARDRAIL PROMPT ---
    # --- THE REFINED LOGIC PROMPT (No Syntax Micromanagement) ---
    # generation.py mein system_prompt ko update karein[cite: 7]:
    system_prompt = (
        "You are an elite, highly disciplined financial auditor. You must follow these strict rules to prevent mathematical errors:\n\n"
        "1. NO MENTAL MATH: You are physically incapable of doing math in your head. NEVER guess a calculation.\n"
        "2. NO COMPLEX FORMULAS: The 'EMI' already includes interest. ALWAYS use the provided EMI. Do not use (P*R*T).\n"
        "3. SINGLE CALCULATIONS ONLY: Break calculations into sequential steps. First calculate Total Payout = (EMI * Tenure). Send this to the calculator. WAIT for the result.\n"
        "4. STRICT CALCULATOR USAGE: When using the calculate tool, only pass the raw mathematical string (e.g., '68690 * 240') to the 'equation' parameter.\n"
        "5. EXTRACT CAREFULLY: Double-check the exact numbers you extract from the text. Ensure no typos.\n"
        "6. Sanctioned Principal MUST be used to calculate total interest paid, NOT the Net Disbursal Amount.\n"
        "7. CRITICAL AWARENESS: You may receive a 'hypothetical document' in your context. NEVER extract numbers or percentages from the hypothetical document. You MUST extract variables ONLY from the actual retrieved legal text."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    print("\n[Agent] Analyzing request...")
    
    # --- THE UPGRADE: 70B Model for the Tool Pass ---
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=[calculator_tool], 
        tool_choice="auto"       
    )

    response_message = response.choices[0].message

    if response_message.tool_calls:
        messages.append(response_message)
        
        for tool_call in response_message.tool_calls:
            if tool_call.function.name == "calculate":
                args = json.loads(tool_call.function.arguments)
                equation = args.get("equation")
                
                print(f"[Agent Tool Execution] Groq requested math: {equation}")
                tool_result = calculate(equation)
                print(f"[Agent Tool Execution] Backend calculated: {tool_result}")
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": tool_result
                })

        print("[Agent] Synthesizing final response...")
        # --- THE UPGRADE: 70B Model for the Final Pass ---
        final_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )
        return final_response.choices[0].message.content

    return response_message.content