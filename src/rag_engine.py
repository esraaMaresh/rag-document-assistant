from google import genai
from typing import List,Dict

SYSTEM_INSTRUCTION = """
You are an expert Document Intellignce Assistant.
Answer ths user's question accurately using ONLY the provided Context Chunks.
Guidelines:
1. Always base your answer strictly on the context. If the answer cannot be found in the context,
explicilty say: "I cannot find information about this in the provided document."
2. Cite the exact page number(s) where you found the information (e.g. "[Page 3]").
3. Keep you answers clear, structured, and factual.
 """

def generate_answer(query: str, context_chunks: List[Dict], api_key: str) -> str:
   """Sends a query + context to Gemini to generate a grounded, cited answer."""
   client = genai.Client(api_key=api_key)

   #format the context block with page metadata
   context_text = "\n\n".join([
      f"--- Context Snippet (Source: {c['source']}, Page: {c['page']}) --- \n{c['content']}"
      for c in context_chunks
   ])


   user_prompt = f"""
Context:
{context_text}

User Question: {query}

Please provide a grounded answer with page citations based on the context above:
"""

 
   response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=user_prompt,
        config={
            "system_instruction": SYSTEM_INSTRUCTION,
            "temperature": 0.2
        }
    )
   return response.text