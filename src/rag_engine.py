import os

# 1. STOP HuggingFace from pinging external servers (kills the 180s timeout!)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict

class VectorStoreManager:
    def __init__(self, collection_name: str = "pdf_rag_collection"):
        # Local embeddings strictly offline
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        # Disable ChromaDB external telemetry pings
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection_name = collection_name
        self._reset_collection()

    def _reset_collection(self):
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )

    def add_chunks(self, chunks: List[Dict], doc_name: str):
        """Adds text chunks with metadata to ChromaDB."""
        self._reset_collection()
        
        documents = [c["content"] for c in chunks]
        metadatas = [{"source": doc_name, "page": c["page"]} for c in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

    def search(self, query: str, top_k: int = 4) -> List[Dict]:
        """Searches for the most semantically relevant chunks."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        formatted_results = []
        if results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            
            for doc, meta in zip(docs, metas):
                formatted_results.append({
                    "content": doc,
                    "page": meta["page"],
                    "source": meta["source"]
                })

        return formatted_results