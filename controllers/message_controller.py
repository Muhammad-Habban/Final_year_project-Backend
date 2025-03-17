# controllers/message_controller.py
import openai
import os
from fastapi import APIRouter, Depends, Query
from repositories.message_repository import MessageRepository
from services.message_service import MessageService
from database import get_database
from models.message import Message

# Load OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

router = APIRouter()

# Dependency Injection for MessageService
def get_message_service(db=Depends(get_database)):
    return MessageService(MessageRepository(db['messages']))

async def generate_llm_response(query: str) -> str:
    """Fetch response from OpenAI API."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use the desired OpenAI model
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": query},
            ]
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

@router.post('/chat', summary="Send query and receive response from LLM", response_model=Message)
async def chat(
    chat_id: str = Query(..., description="Chat session ID"),
    query: str = Query(..., description="User input query"),
    message_service: MessageService = Depends(get_message_service),
):
    user_id = "user_bot"
    
    # Get response from OpenAI
    llm_response = await generate_llm_response(query)

    # Create and store the message using the service
    message = await message_service.create_message(chat_id, user_id, query, llm_response)

    return message
