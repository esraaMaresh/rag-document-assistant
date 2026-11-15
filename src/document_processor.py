from pypdf import PdfReader
from typing import List, Dict

def extract_text_from_pdf(pdf_file) -> List[Dict]:
    """
    Extracts text from a PDF file object or path, preserving page numbers.
    Returns a list of dicts: [{'page': 1, 'text': '...'}]
    """
    reader = PdfReader(pdf_file)
    pages_data = []
    
    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages_data.append({
                "page": page_idx + 1,
                "text": text.strip()
            })
            
    return pages_data

def chunk_text(pages_data: List[Dict], chunk_size: int = 600, chunk_overlap: int = 100) -> List[Dict]:
    """
    Splits text into overlapping chunks while tracking page metadata.
    """
    chunks = []
    
    for page_info in pages_data:
        text = page_info["text"]
        page_num = page_info["page"]
        
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_content = text[start:end]
            
            if chunk_content.strip():
                chunks.append({
                    "content": chunk_content.strip(),
                    "page": page_num
                })
                
            start += (chunk_size - chunk_overlap)
            
    return chunks