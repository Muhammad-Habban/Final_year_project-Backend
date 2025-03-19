from fastapi import APIRouter, Depends, HTTPException, Query
from repositories.message_repository import MessageRepository
from services.message_service import MessageService
from database import get_database
from models.message import Message

router = APIRouter()

def get_message_service(db=Depends(get_database)):
    return MessageService(MessageRepository(db['messages']))

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
