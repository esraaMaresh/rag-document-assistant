import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.document_processor import extract_text_from_pdf, chunk_text
from src.vector_store import VectorStoreManager
from src.rag_engine import generate_answer

st.set_page_config(
    page_title="RAG Document Assistant", 
    page_icon=":material/article:", 
    layout="wide"
)

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStoreManager()
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Check Streamlit Cloud Secrets First ---
secret_key = st.secrets.get("GEMINI_API_KEY", None)

# Sidebar: Config & PDF Upload
with st.sidebar:
    st.header(":material/tune: Configuration")
    
    if secret_key:
        api_key = secret_key
        st.success(":material/verified: Connected & Ready")
    else:
        # Fallback only if running on a machine without Secrets configured
        api_key = st.text_input(
            "Gemini API Key", 
            type="password", 
            help="Get a free key at aistudio.google.com"
        )
    
    st.markdown("---")
    st.header(":material/upload_file: Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

    if uploaded_file and not st.session_state.doc_processed:
        with st.spinner("Processing & indexing document..."):
            pages_data = extract_text_from_pdf(uploaded_file)
            chunks = chunk_text(pages_data)
            st.session_state.vector_store.add_chunks(chunks, uploaded_file.name)
            st.session_state.doc_processed = True
            st.success(f":material/check_circle: Indexed {len(chunks)} chunks across {len(pages_data)} pages!")

    if uploaded_file is None:
        st.session_state.doc_processed = False

# Main Chat Interface
st.title(":material/smart_toy: Document Intelligence & Q&A Assistant")
st.markdown("Upload any PDF report, contract, or paper to ask questions with **verifiable page citations**.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(":material/menu_book: View Source Citations"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**:material/bookmark: Source {idx} (Page {src['page']}):**")
                    st.caption(src["content"])

# User question input
if prompt := st.chat_input("Ask a question about your document..."):
    if not api_key:
        st.error(":material/key_off: Please enter your Gemini API Key in the left sidebar first!")
    elif not st.session_state.doc_processed:
        st.warning(":material/warning: Please upload a PDF document in the sidebar first!")
    else:
        # Display user question
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve relevant chunks
        with st.spinner("Searching document & generating answer..."):
            relevant_chunks = st.session_state.vector_store.search(prompt, top_k=4)
            
            try:
                answer = generate_answer(prompt, relevant_chunks, api_key)
                
                # Display response
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    with st.expander(":material/menu_book: View Source Citations"):
                        for idx, src in enumerate(relevant_chunks, 1):
                            st.markdown(f"**:material/bookmark: Source {idx} (Page {src['page']}):**")
                            st.caption(src["content"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": relevant_chunks
                })
            except Exception as e:
                st.error(f":material/error: Error generating answer: {str(e)}")