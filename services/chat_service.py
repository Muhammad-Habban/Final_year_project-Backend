from repositories.chat_repository import ChatRepository
from uuid import uuid4
from datetime import datetime
import os
from unstructured.partition.text import partition_text
from unstructured.partition.auto import partition
from sentence_transformers import SentenceTransformer
import aiofiles
import json

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
        embeddings = []
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
            embedding = model.encode(element.text).tolist()
            embeddings.append(embedding)
            
            # Update the page number if a page break is detected
            if "page_break" in str(element):  # Check if the element indicates a page break
                current_page_number += 1

        # Create directory if not exists
        chunks_dir = "chunks"
        os.makedirs(chunks_dir, exist_ok=True)
        os.makedirs("embeddings", exist_ok=True)
        
        # Define file path
        file_name = f"{chat_id}.db"
        file_path = os.path.join(chunks_dir, file_name)
        
        # Write chunks to file
        async with aiofiles.open(file_path, 'w') as file:
            for chunk in chunks:
                await file.write(str(chunk) + "\n")

        # Write embeddings to file
        embeddings_path = os.path.join("embeddings", f"{chat_id}.db")
        async with aiofiles.open(embeddings_path, 'w') as file:
            for embedding in embeddings:
                await file.write(json.dumps(embedding) + "\n")
        
        

