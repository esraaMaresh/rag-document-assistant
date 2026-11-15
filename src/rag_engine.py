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

# Current high-availability models (Flash-Lite handles high-demand traffic best)
MODELS_TO_TRY = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.6-flash"
]

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
        # Try each model up to 2 times if temporary high demand
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config={
                        "system_instruction": SYSTEM_INSTRUCTION
                    }
                )
                return response.text
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                # If 503, 429, or temporary high demand, wait briefly and try next
                if "503" in err_str or "high demand" in err_str or "429" in err_str:
                    time.sleep(1.5)
                    continue
                else:
                    break

    raise last_error