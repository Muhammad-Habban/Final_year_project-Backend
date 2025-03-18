import json
from models.message import Message
from pymongo.collection import Collection
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class MessageRepository:
    def __init__(self, db: Collection, faiss_index_path: str):
        self.db = db
        self.faiss_index_path = faiss_index_path
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Load FAISS index
        self.index = faiss.read_index(self.faiss_index_path)

    async def get_chunks_for_query(self, query: str, top_k: int = 20):
        query_embedding = self.model.encode(query).reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, top_k)

        similar_chunks = []
        for idx in indices[0]:
            if idx == -1:
                continue
            chunk = await self.db.find_one({"id": idx})
            if chunk:
                similar_chunks.append({
                    "id": chunk["_id"],
                    "type": chunk["type"],
                    "text": chunk["text"],
                    "page_number": chunk["page_number"]
                })
        return similar_chunks

    def hybrid_search(self, query, top_k=20, min_words=5, exclude_types=[]):
        # Step 1: FAISS-based semantic search
        faiss_results = self.get_chunks_for_query(query, top_k)

        # Step 2: BM25-based keyword search
        corpus = [chunk["text"] for chunk in faiss_results]
        bm25 = BM25Okapi(corpus)
        tokenized_query = query.split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[-top_k:][::-1]

        bm25_results = []
        for idx in bm25_indices:
            if idx < len(faiss_results):
                chunk = faiss_results[idx]
                bm25_results.append(chunk)

        # Step 3: Combine results and filter
        combined_results = faiss_results + bm25_results
        unique_results = {res["id"]: res for res in combined_results}.values()
        filtered_results = [res for res in unique_results if len(res["text"].split()) >= min_words and res["type"] not in exclude_types]

        return sorted(filtered_results, key=lambda x: len(x["text"].split()), reverse=True)[:top_k]

    async def create_message(self, chat_id: str, user_id: str, text: str, response: str):
        message = Message.create(chat_id, user_id, text, response)
        message_dict = message.dict()
        await self.db.insert_one(message_dict)
        return message

    async def get_messages_by_chat_id(self, chat_id: str):
        messages = self.db.find({"chat_id": chat_id})
        return [msg async for msg in messages]

    async def get_all_messages(self):
        messages = self.db.find({})
        return [msg async for msg in messages]
