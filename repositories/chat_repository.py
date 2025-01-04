from models.chat import Chat
from pymongo.collection import Collection

class ChatRepository:
    def __init__(self, db: Collection):
        self.db = db

    async def create_chat(self, chat: Chat):
        chat = dict(chat)
        await self.db.insert_one(chat)
        return chat
