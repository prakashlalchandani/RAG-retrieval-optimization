import json
import ast
import operator

def calculate(equation: str) -> str:
    """A highly secure mathematical evaluator for the LLM to use."""
    try:
        # We use ast.literal_eval for basic math security instead of the dangerous raw eval()
        # For a simple calculator, we parse the math string and calculate it safely
        def eval_math(node):
            operators = {ast.Add: operator.add, ast.Sub: operator.sub, 
                         ast.Mult: operator.mul, ast.Div: operator.truediv}
            if isinstance(node, ast.Num): # <number>
                return node.n
            elif isinstance(node, ast.BinOp): # <left> <operator> <right>
                return operators[type(node.op)](eval_math(node.left), eval_math(node.right))
            else:
                raise TypeError(node)

        result = eval_math(ast.parse(equation, mode='eval').body)
        return str(result)
    except Exception as e:
        return f"Error calculating: Invalid equation format. {e}"

# This is the JSON Schema that teaches Groq how to use your Python function
calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Calculates the exact result of a mathematical equation. MUST be used whenever the user asks for total loan costs, interest differences, or financial multiplications.",
        "parameters": {
            "type": "object",
            "properties": {
                "equation": {
                    "type": "string",
                    "description": "The mathematical equation to evaluate (e.g., '(32425 * 60) - 1500000'). Do NOT include currency symbols like 'Rs.' or commas in the equation."
                }
            },
            "required": ["equation"]
        }
    }
}