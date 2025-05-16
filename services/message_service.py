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

    async def create_message(self, chat_id: str, text: str, response: str):
        return await self.message_repository.create_message(chat_id, text, response)

    async def get_message_by_id(self, message_id: str):
        return await self.message_repository.get_message_by_id(message_id)

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
            raise FileNotFoundError(
                f"FAISS index file for chat_id {chat_id} not found.")

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

    # New helper method
    def build_chunk_dict(self, row):
        return {
            "id": row[0],
            "type": row[1],
            "text": row[2],
            "page_number": row[3],
            "section_number": row[4],
            "chapter_number": row[5]
        }

    # Updated search_similar_chunks: no filtering
    def search_similar_chunks(self, chat_id: str, query: str, top_k=50):
        index = self.load_faiss_index(chat_id)
        query_vector = self.model.encode(query).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        distances, indices = index.search(query_vector, top_k)

        results = []
        chunks_db_path = os.path.join("chunks", f"{chat_id}.db")
        conn = sqlite3.connect(chunks_db_path)
        cursor = conn.cursor()

        for faiss_index in indices[0]:
            if faiss_index == -1:
                continue
            cursor.execute(
                "SELECT id FROM chunks LIMIT 1 OFFSET ?", (int(faiss_index),))
            row_id = cursor.fetchone()
            if row_id:
                cursor.execute("""
                    SELECT id, type, text, page_number, section_number, chapter_number
                    FROM chunks
                    WHERE id = ?
                """, (row_id[0],))
                row = cursor.fetchone()
                if row:
                    results.append(self.build_chunk_dict(row))
        return results

    # Refactor hybrid_search to apply all filtering and sorting here only
    def hybrid_search(self, chat_id, query, top_k=20, exclude_types=[]):
        # Step 1: FAISS Search
        faiss_results = self.search_similar_chunks(chat_id, query, top_k=50)
        if not faiss_results:
            raise HTTPException(
                status_code=400, detail="No results found from FAISS search.")

        # Step 2: BM25 Scoring on FAISS Text Corpus
        corpus = [chunk["text"] for chunk in faiss_results]
        bm25 = BM25Okapi(corpus)
        tokenized_query = query.split()
        bm25_scores = bm25.get_scores(tokenized_query)

        # Step 3: Rank FAISS chunks by BM25 score
        bm25_indices = np.argsort(
            bm25_scores)[-top_k:][::-1]  # Top BM25 indices
        bm25_ranked_results = [faiss_results[i]
                               for i in bm25_indices if i < len(faiss_results)]

        # Step 4: Combine FAISS and BM25-ranked results
        combined_results = faiss_results + bm25_ranked_results

        # Step 5: Deduplicate by `id`
        unique_results = []
        seen_ids = set()
        for result in combined_results:
            if result["id"] not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result["id"])

        # Step 6: Apply chunk type filters
        filtered_results = self.filter_chunks_by_type(
            unique_results, exclude_types)

        # Step 7: Sort by length (or skip if you want raw relevance)
        sorted_results = self.sort_chunks_by_length(filtered_results)

        # Step 8: Ensure exactly top_k returned
        final_results = []
        for chunk in sorted_results[:top_k]:
            final_results.append({
                "page_number": chunk.get("page_number"),
                "text": chunk.get("text"),
                "chapter_number": chunk.get("chapter_number"),
                "section_number": chunk.get("section_number")
            })

        return final_results
