import openai
import os
from fastapi import APIRouter, Depends, HTTPException, Query
from repositories.message_repository import MessageRepository
from services.message_service import MessageService
from database import get_database
from models.message import Message

# Load OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

router = APIRouter()

def get_message_service(db=Depends(get_database)):
    faiss_index_path = "path_to_your_faiss_index_file"  # Update with the correct path
    return MessageService(MessageRepository(db['messages'], faiss_index_path))

# Route to send query and receive response from LLM
@router.post('/chat', summary="Send query and receive response from LLM", response_model=Message)
async def chat(
    chat_id: str = Query(..., description="Chat session ID"),
    query: str = Query(..., description="User input query"),
    message_service: MessageService = Depends(get_message_service),
):
    user_id = "user_bot"  # You may modify this depending on the user authentication
    message = await message_service.get_response_for_query(chat_id, user_id, query)
    return message

# Route to get all messages in the system
@router.get("/messages", tags=["messages"], summary="Get all messages", response_model=list[Message])
async def get_all_messages(message_service: MessageService = Depends(get_message_service)):
    try:
        messages = await message_service.get_all_messages()
        return messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching messages: {str(e)}")

# Route to get messages by chat_id
@router.get("/messages/{chat_id}", tags=["messages"], summary="Get messages by chat ID", response_model=list[Message])
async def get_messages_by_chat_id(
    chat_id: str,
    message_service: MessageService = Depends(get_message_service)
):
    try:
        messages = await message_service.get_messages_by_chat_id(chat_id)
        return messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching messages: {str(e)}")
