from google import genai
from typing import List, Dict
import time

SYSTEM_INSTRUCTION = """
You are an expert Document Intelligence Assistant. 
Answer the user's question accurately using ONLY the provided Context Chunks.
Guidelines:
1. Always base your answer strictly on the context. If the answer cannot be found in the context, explicitly say: "I cannot find information about this in the provided document."
2. Cite the exact page number(s) where you found the information (e.g. "[Page 3]").
3. Keep your answers clear, structured, and factual.
"""

# Resilient list of models to try in order
MODELS_TO_TRY = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-1.5-flash"]

def generate_answer(query: str, context_chunks: List[Dict], api_key: str) -> str:
    """Sends query + context to Gemini with automatic model fallback."""
    client = genai.Client(api_key=api_key)

    context_text = "\n\n".join([
        f"--- Context Snippet (Source: {c['source']}, Page: {c['page']}) ---\n{c['content']}"
        for c in context_chunks
    ])

    user_prompt = f"""
Context:
{context_text}

User Question: {query}

Please provide a grounded answer with page citations based on the context above:
"""

    last_error = None
    for model_name in MODELS_TO_TRY:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config={
                    "system_instruction": SYSTEM_INSTRUCTION,
                    "temperature": 0.2
                }
            )
            return response.text
        except Exception as e:
            last_error = e
            # If server is overloaded (503) or rate-limited (429), smoothly try the backup model
            if "503" in str(e) or "high demand" in str(e).lower() or "429" in str(e):
                time.sleep(1)
                continue
            raise e

    # If all models failed, raise the final error
    raise last_error