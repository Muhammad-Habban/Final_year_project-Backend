from llama_cpp import Llama
import json
import re

# Load GGUF model
model_path = "E:\\deepseek-llm-7b-chat.Q4_K_M.gguf"
llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)

def generate_quiz(context: str):
    prompt = f"""
    Given the following text, create exactly 5 multiple-choice quiz questions.
    
    Strictly return ONLY JSON. Do NOT include any explanation, comments, or additional text outside the JSON.
    
    JSON format example:
    {{
      "questions": [
        {{
          "description": "Question text?",
          "options": ["Option A", "Option B", "Option C", "Option D"],
          "answer": "Correct Option"
        }}
      ]
    }}

    Text to use:
    {context}

    JSON:
    """

    try:
        output = llm(
            prompt,
            max_tokens=1000,
            temperature=0.5,
            stop=["\n\n"]
        )
        response_text = output["choices"][0]["text"].strip()

        # Extract JSON from the response using regex (robust method)
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in the LLM response.")

        json_content = json_match.group(0)

        quiz_json = json.loads(json_content)
        return quiz_json

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON returned by LLM. Error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {str(e)}")


# Example usage
context = """
The quick brown fox jumps over the lazy dog. This is a classic pangram that contains every letter of the English alphabet. It is often used to test fonts and keyboards. The phrase has become a part of popular culture and is frequently referenced in various media. The quick brown fox is often depicted as clever and agile, while the lazy dog represents a more laid-back attitude. Together, they create a vivid image of contrast between speed and sloth.
"""

quiz = generate_quiz(context)
print(json.dumps(quiz, indent=2))
