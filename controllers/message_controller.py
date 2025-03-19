from fastapi import APIRouter, Depends, HTTPException, Query
from repositories.message_repository import MessageRepository
from services.message_service import MessageService
from database import get_database
from models.message import Message
import openai
import os
from models.request_body import RequestBody

# Load OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

router = APIRouter()


def get_message_service(db=Depends(get_database)):
    return MessageService(MessageRepository(db['messages']))

# Route to get all messages in the system
@router.post("/getresponse", tags=["LLM"], summary="Send user prompt to LLM and save response")
async def get_response(
    request: RequestBody,  
    message_service: MessageService = Depends(get_message_service),
):
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

@router.post("/getresponse", tags=["LLM"], summary="Send user prompt to GPT-3.5-turbo and save response")
async def get_response(
    chat_id: str = Query(..., description="Chat session ID"),
    user_id: str = Query(..., description="User ID"),
    user_prompt: str = Query(..., description="User input prompt for LLM"), 
    message_service: MessageService = Depends(get_message_service),
):
    try:
        # Send the user prompt to GPT-3.5-turbo via the Chat Completions API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        
        # Extract the response text
        llm_response = response.choices[0].message.content.strip()
        
        # Create and save the message with the user prompt and LLM response
        message = await message_service.create_message(
            chat_id=chat_id,
            user_id=user_id,
            text=user_prompt,
            response=llm_response
        )
        
        # Return the saved message with the response
        return message
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")










