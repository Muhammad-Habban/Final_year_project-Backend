# repositories/message_repository.py
from models.message import Message
from pymongo.collection import Collection

class MessageRepository:
    def __init__(self, db: Collection):
        self.db = db
        
    async def create_message(self, chat_id: str, user_id: str, text: str, response: str):
        message = Message.create(chat_id, user_id, text, response)
        message_dict = message.dict()
        await self.db.insert_one(message_dict)
        return message

    async def get_messages_by_chat_id(self, chat_id: str):
        messages = self.db.find({"chat_id": chat_id})
        return [msg async for msg in messages]
