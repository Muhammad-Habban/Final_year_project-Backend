from models.chat import Chat
from pymongo.collection import Collection

class ChatRepository:
    def __init__(self, collection):
        self.collection = collection

    async def create_chat(self, chat):
        await self.collection.insert_one(chat)

    async def update_chat_path(self, chat_id: str, path: str):
        await self.collection.update_one({"chat_id": chat_id}, {"$set": {"path": path}})
    
    async def get_all_chats(self):
        try:
            chats = await self.collection.find().to_list(None)
            return chats
        except Exception as e:
            print(e)
            return None

    async def get_chats_by_user_id(self, user_id: str):
        chats = await self.collection.find({"user_id": user_id}).to_list(None)
        return chats

    async def get_chat_by_id(self, chat_id: str):
        chat = await self.collection.find_one({"chat_id": chat_id})
        return chat

    async def update_chat(self, chat_id: str, updated_fields: dict):
        await self.collection.update_one({"chat_id": chat_id}, {"$set": updated_fields})

    async def delete_chat(self, chat_id: str):
        await self.collection.delete_one({"chat_id": chat_id})