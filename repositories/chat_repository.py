from models.chat import Chat
from pymongo.collection import Collection

class ChatRepository:
    def __init__(self, collection):
        self.collection = collection

    async def create_chat(self, chat):
        await self.collection.insert_one(chat)

    async def update_chat_path(self, chat_id: str, path: str):
        await self.collection.update_one({"chat_id": chat_id}, {"$set": {"path": path}})
