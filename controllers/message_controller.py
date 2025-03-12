# controllers/message_controller.py
from fastapi import APIRouter, Depends, Query
from repositories.message_repository import MessageRepository
from services.message_service import MessageService
from database import get_database
from models.message import Message

router = APIRouter()

# Dependency Injection for MessageService
def get_message_service(db=Depends(get_database)):
    return MessageService(MessageRepository(db['messages']))

@router.post('/chat', summary="Send query and receive response from LLM", response_model=Message)
async def chat(
    chat_id: str = Query(..., description="Chat session ID"),
    query: str = Query(..., description="User input query"),
    message_service: MessageService = Depends(get_message_service),
):
    user_id = "user_bot"
    
    # Here you would integrate actual LLM/OpenAI logic
    llm_response = f"LLM generated response to query '{query}'"

    # Create and store the message using the service
    message = await message_service.create_message(chat_id, user_id, query, llm_response)

    return message
