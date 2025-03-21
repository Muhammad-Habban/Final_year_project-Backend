import openai
import os
from fastapi import APIRouter, HTTPException, Query, Depends
from repositories.message_repository import MessageRepository
from services.message_service import MessageService
from database import get_database
from models.message import Message
import os
import torch
import numpy as np
from fastapi import APIRouter, HTTPException, File, UploadFile
from transformers import pipeline

# Load OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
router = APIRouter()

def get_message_service(db=Depends(get_database)):
    return MessageService(MessageRepository(db['messages']))


@router.post("/convert_audio", tags=["LLM"], summary="Convert MP3 audio file to text using Whisper")
async def convert_audio_to_text(file: UploadFile = File(...)):
    """Receive an MP3 file, convert it to text using Whisper."""
    try:
        # Convert the uploaded MP3 to WAV
        wav_audio = convert_mp3_to_wav(file)

        # Read the WAV data and transcribe using Whisper
        audio_data = np.frombuffer(wav_audio.read(), dtype=np.float32)

        # Perform speech-to-text using Whisper
        transcription = whisper_pipeline({"sampling_rate": SAMPLE_RATE, "raw": audio_data}, generate_kwargs={"language": "en"})['text']

        # Return the transcription result
        return {"transcription": transcription}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio file: {str(e)}")
    

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


@router.post("/getresponse", tags=["LLM"], summary="Send user prompt to GPT-4 and save response")
async def get_response(
    chat_id: str = Query(..., description="Chat session ID"),
    user_id: str = Query(..., description="User ID"),
    user_prompt: str = Query(..., description="User input prompt for LLM"),
    message_service: MessageService = Depends(get_message_service),
):

    try:
        # Send the user prompt to GPT-4 model via the correct chat completions endpoint
        response = openai.ChatCompletion.create(  # Correct usage of ChatCompletion.create
            model="gpt-4",  # Use GPT-4 model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        # Extract the response text
        llm_response = response['choices'][0]['message']['content'].strip()

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


# Endpoint for feeding chunks to LLM and getting response
@router.post("/test_chunks", tags=["LLM"], summary="Perform hybrid search and send result to GPT-4")
async def test_chunks(
    chat_id: str = Query(..., description="Chat session ID"),
    user_prompt: str = Query(..., description="User input prompt for LLM"),
    message_service: MessageService = Depends(get_message_service),
):
    try:
        # Perform hybrid search to retrieve relevant chunks using the hybrid_search method
        faiss_results = message_service.hybrid_search(query=user_prompt,chat_id=chat_id, top_k=5, min_words=5)

        # Prepare the chunks for the prompt (you can adjust how many chunks you want to include)
        combined_chunks = " ".join([chunk["text"] for chunk in faiss_results])  # Combine the top chunks

        # Concatenate the user prompt with the relevant chunks
        full_prompt = combined_chunks + "\n" + user_prompt

        # Send the full prompt to GPT-4 via the OpenAI API
        response = openai.ChatCompletion.create(  # Correct usage of ChatCompletion.create
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt},
            ],
            max_tokens=1000,
            temperature=0.7
        )

        # Extract the response text
        llm_response = response['choices'][0]['message']['content'].strip()

        # Create and save the message with the user prompt and LLM response
        message = await message_service.create_message(
            chat_id=chat_id,
            user_id="user_id_placeholder",  # Replace with actual user ID if available
            text=user_prompt,
            response=llm_response
        )

        # Return the saved message with the response
        return message

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chunks and generating response: {str(e)}")
