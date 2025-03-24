from repositories.message_repository import MessageRepository
from typing import List
from fastapi import HTTPException
from sentence_transformers import SentenceTransformer
import sqlite3
import faiss
import json
import os
from rank_bm25 import BM25Okapi
import numpy as np

class MessageService:
    def __init__(self, message_repository: MessageRepository):
        self.message_repository = message_repository
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    async def create_message(self, chat_id: str,text: str, response: str):
        return await self.message_repository.create_message(chat_id, text, response)
    
    async def get_messages_by_chat_id(self, chat_id: str):
        return await self.message_repository.get_messages_by_chat_id(chat_id)

    async def get_all_messages(self):
        return await self.message_repository.get_all_messages()
    
    def load_faiss_index(self, chat_id: str):
        """Load FAISS index for a specific chat_id from the corresponding .bin file."""
        faiss_dir = "faiss"
        faiss_file_path = os.path.join(faiss_dir, f"{chat_id}.bin")

        if os.path.exists(faiss_file_path):
            # Load the FAISS index from the file
            index = faiss.read_index(faiss_file_path)
            return index
        else:
            raise FileNotFoundError(f"FAISS index file for chat_id {chat_id} not found.")
    
    def filter_chunks_by_word_count(self, chunks, min_words=5):
        """Filter out chunks that have fewer than `min_words` words."""
        filtered_chunks = []
        for chunk in chunks:
            word_count = len(chunk["text"].split())  # Count words in the chunk
            if word_count >= min_words:
                filtered_chunks.append(chunk)
        return filtered_chunks

    def filter_chunks_by_type(self, chunks, exclude_types=["title", "heading"]):
        """Filter out chunks of specific types."""
        filtered_chunks = []
        for chunk in chunks:
            if chunk["type"] not in exclude_types:
                filtered_chunks.append(chunk)
        return filtered_chunks

    def sort_chunks_by_length(self, chunks):
        """Sort chunks by word count in descending order."""
        return sorted(chunks, key=lambda x: len(x["text"].split()), reverse=True)

    def search_similar_chunks(self, chat_id: str, query: str, top_k=50):
        """Search for the most similar chunks using the FAISS index for the specific chat_id."""
        
        # Load FAISS index for the given chat_id
        index = self.load_faiss_index(chat_id)

        # Step 1: Convert query to embedding
        query_vector = self.model.encode(query).reshape(1, -1)
        
        # Normalize the query embedding
        faiss.normalize_L2(query_vector)
        
        # Step 2: Use FAISS to find the nearest embeddings
        distances, indices = index.search(query_vector, top_k)  # FAISS returns cosine similarity scores

        # Step 3: Retrieve corresponding chunks from SQLite
        results = []
        chunks_db_path = os.path.join("chunks", f"{chat_id}.db")
        conn = sqlite3.connect(chunks_db_path)  # Connect to the appropriate SQLite DB
        cursor = conn.cursor()

        for faiss_index in indices[0]:  # FAISS returns indices
            if faiss_index == -1:
                continue  # Skip invalid index
            
            # Fetch the actual database row ID corresponding to FAISS index
            cursor.execute("SELECT id FROM chunks LIMIT 1 OFFSET ?", (int(faiss_index),))
            row_id = cursor.fetchone()
            if row_id:
                cursor.execute("SELECT id, type, text, page_number FROM chunks WHERE id=?", (row_id[0],))
                row = cursor.fetchone()
                if row:
                    results.append({
                        "id": row[0],
                        "type": row[1],
                        "text": row[2],
                        "page_number": row[3]
                    })
        
        results = self.filter_chunks_by_word_count(results)
        results = self.filter_chunks_by_type(results)
        return results

    ### Hybrid Search
    # Perform hybrid search
    def hybrid_search(self,chat_id, query, top_k=20, min_words=5, exclude_types=[]):
        """Retrieve top_k most similar chunks using hybrid search and filter by word count and type."""
        
        # Create a BM25 index for keyword-based search
        faiss_results = self.search_similar_chunks(chat_id, query, top_k=50)
        corpus = [chunk["text"] for chunk in faiss_results]  # Make sure 'faiss_results' is defined or passed to this method
        bm25 = BM25Okapi(corpus)

        # Step 1: Semantic search with FAISS
        
        # Ensure that faiss_results is valid and contains chunks
        if not faiss_results:
            raise HTTPException(status_code=400, detail="No results found from FAISS search.")

        # Step 2: Keyword search with BM25
        tokenized_query = query.split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[-top_k:][::-1]  # Get top_k indices
        
        # Step 3: Fetch BM25 results with the same structure as FAISS results
        bm25_results = []
        for idx in bm25_indices:
            if idx < len(faiss_results):  # Ensure the index is within bounds
                chunk = faiss_results[idx]
                bm25_results.append({
                    "id": idx,  # Use the index as the ID (or fetch the actual ID from the database if needed)
                    "type": chunk["type"],
                    "text": chunk["text"],
                    "page_number": chunk["page_number"]
                })
        
        # Step 4: Combine results
        combined_results = faiss_results + bm25_results
        
        # Step 5: Remove duplicates (if any)
        unique_results = []
        seen_ids = set()
        for result in combined_results:
            if result["id"] not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result["id"])
        
        # Step 6: Filter chunks by word count and type
        filtered_results = self.filter_chunks_by_word_count(unique_results, min_words=min_words)
        filtered_results = self.filter_chunks_by_type(filtered_results, exclude_types=exclude_types)
        
        # Step 7: Sort chunks by length and return top_k
        sorted_results = self.sort_chunks_by_length(filtered_results)
        return sorted_results[:top_k]  # Return exactly top_k chunks

