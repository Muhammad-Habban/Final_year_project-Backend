from repositories.chat_repository import ChatRepository
from uuid import uuid4
from datetime import datetime
import os
from unstructured.partition.text import partition_text
from unstructured.partition.auto import partition
from sentence_transformers import SentenceTransformer
import aiofiles
import json
import sqlite3
import faiss
import numpy as np

class ChatService:
    def __init__(self, chat_repository: ChatRepository):
        self.chat_repository = chat_repository

    async def create_chat(self, user_id: str, title: str):
        chat_id = str(uuid4())
        created_at = datetime.now().isoformat()
        updated_at = created_at
        chat = {"chat_id": chat_id, "user_id": user_id, "title": title, "created_at": created_at, "updated_at": updated_at}
        
        await self.chat_repository.create_chat(chat)
        return chat

    async def update_chat_path(self, chat_id: str, path: str):
        await self.chat_repository.update_chat_path(chat_id, path)

    async def create_chunks(self, chat_id: str, path: str):
        # Use Unstructured to partition the combined text into chunks
        # elements = partition_text(path)
        elements = partition(path, content_type="application/pdf")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        # embeddings = []
        # Create chunks with page numbers
        chunks = []
        current_page_number = 1  # Start with page 1

        for element in elements:
            # Assign the current page number to the chunk
            chunk_data = {
                'type': element.__class__.__name__,
                'text': element.text,
                'page_number': current_page_number  # Assign the page number
            }
            chunks.append(chunk_data)
            # embedding = model.encode(element.text).tolist()
            # embeddings.append(embedding)
            
            # Update the page number if a page break is detected
            if "page_break" in str(element):  # Check if the element indicates a page break
                current_page_number += 1

        # Create directory if not exists
        chunks_dir = "chunks"
        os.makedirs(chunks_dir, exist_ok=True)
        # os.makedirs("embeddings", exist_ok=True)
        for chunk in chunks:
            chunk["embedding"] = model.encode(chunk["text"]).tolist()  # Convert numpy array to list for storage
        # Define file path
        file_name = f"{chat_id}.db"
        file_path = os.path.join(chunks_dir, file_name)
        # Connect to SQLite database (or create one if it doesn't exist)
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()

        # Create table to store chunks and embeddings
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT,
            text TEXT,
            page_number INTEGER,
            embedding TEXT  -- Store as JSON string
        )
        """)

        # Insert chunk data into database
        for chunk in chunks:
            cursor.execute("""
            INSERT INTO chunks (type, text, page_number, embedding) VALUES (?, ?, ?, ?)
            """, (chunk["type"], chunk["text"], chunk["page_number"], json.dumps(chunk["embedding"])))  # Store embedding as JSON

        # Commit and close
        conn.commit()
        conn.close()

        embedding_list = []

        ids = []

        connection = sqlite3.connect(file_path)  # Reopen the connection if it was closed
        cursor = connection.cursor()

        cursor.execute("SELECT id, embedding FROM chunks")
        rows = cursor.fetchall()

        for row in rows:
            chunk_id = row[0]
            embedding = json.loads(row[1])  # Convert JSON string back to a list
            embedding_list.append(embedding)
            ids.append(chunk_id)

        # Convert to numpy arrays
        embedding_array = np.array(embedding_list, dtype=np.float32)
        ids_array = np.array(ids, dtype=np.int64)  # Store actual chunk IDs

        # Normalize embeddings before indexing
        faiss.normalize_L2(embedding_array)  # Normalize the embeddings
        # Initialize FAISS index
        embedding_dimension = embedding_array.shape[1]
        index = faiss.IndexFlatIP(embedding_dimension)

        # Create ID-based FAISS index
        index_with_ids = faiss.IndexIDMap(index)
        index_with_ids.add_with_ids(embedding_array, ids_array)  # Store IDs inside FAISS

        faiss_dir = "faiss"
        os.makedirs(faiss_dir, exist_ok=True)

        # Define file path
        faiss_file_name = f"{chat_id}.bin"
        faiss_file_path = os.path.join(faiss_dir, faiss_file_name)

        # Save FAISS index
        faiss.write_index(index_with_ids, f"{faiss_file_path}")

        index = faiss.read_index(f"{faiss_file_path}")
        
        
    async def get_all_chats(self):
        chats = await self.chat_repository.get_all_chats()
        return [
            {**chat, "_id": str(chat["_id"])} for chat in chats
        ]
        

    async def get_chats_by_user_id(self, user_id: str):
        chats = await self.chat_repository.get_chats_by_user_id(user_id)
        return [
            {**chat, "_id": str(chat["_id"])} for chat in chats
        ]

    async def get_chat_by_id(self, chat_id: str):
        chat = await self.chat_repository.get_chat_by_id(chat_id)
        return chat

    async def update_chat(self, chat_id: str, updated_fields: dict):
        await self.chat_repository.update_chat(chat_id, updated_fields)

    async def delete_chat(self, chat_id: str):
        await self.chat_repository.delete_chat(chat_id)