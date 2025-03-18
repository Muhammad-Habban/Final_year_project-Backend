from repositories.message_repository import MessageRepository
import openai
from fastapi import HTTPException

class MessageService:
    def __init__(self, message_repository: MessageRepository):
        self.message_repository = message_repository

    async def create_message(self, chat_id: str, user_id: str, text: str, response: str):
        return await self.message_repository.create_message(chat_id, user_id, text, response)

    async def get_messages_by_chat_id(self, chat_id: str):
        return await self.message_repository.get_messages_by_chat_id(chat_id)

    async def get_all_messages(self):
        return await self.message_repository.get_all_messages()

    async def generate_llm_response(self, query: str, chunks: list) -> str:
        context = "\n".join([chunk["text"] for chunk in chunks])

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in physics."},
                    {"role": "user", "content": f"Context: {context}\nQuery: {query}"}
                ]
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

    async def get_response_for_query(self, chat_id: str, user_id: str, query: str):
        # Perform hybrid search to retrieve the most relevant chunks
        similar_chunks = self.message_repository.hybrid_search(query)
        llm_response = await self.generate_llm_response(query, similar_chunks)
        message = await self.create_message(chat_id, user_id, query, llm_response)
        return message
