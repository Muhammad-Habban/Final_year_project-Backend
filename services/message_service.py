# services/message_service.py
from repositories.message_repository import MessageRepository

class MessageService:
    def __init__(self, message_repository: MessageRepository):
        self.message_repository = message_repository
        
    async def create_message(self, chat_id: str, user_id: str, text: str, response: str):
        return await self.message_repository.create_message(chat_id, user_id, text, response)

    async def get_messages_by_chat_id(self, chat_id: str):
        return await self.message_repository.get_messages_by_chat_id(chat_id)