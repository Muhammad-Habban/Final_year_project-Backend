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
from fastapi import File, UploadFile
from transformers import pipeline
from ollama import chat 
import json
import re
from llama_cpp import Llama
# Load OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
router = APIRouter()

# Load GGUF quantized model
# model_path = "E:\\deepseek-llm-7b-chat.Q4_K_M.gguf"
# llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)

def get_message_service(db=Depends(get_database)):
    return MessageService(MessageRepository(db['messages']))



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
        
        print("Combined Chunks:", combined_chunks)

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
            text=user_prompt,
            response=llm_response
        )

        # Return the saved message with the response
        return message

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chunks and generating response: {str(e)}")


# Getting response from DeepSeek
@router.post("/deepseek_response", tags=["LLM"], summary="Send user prompt to Deepseek model and save response")
async def get_deepseek_response(
    chat_id: str = Query(..., description="Chat session ID"),
    user_prompt: str = Query(..., description="User input prompt for Deepseek model"),
    message_service: MessageService = Depends(get_message_service),
):
    try:
        response = chat(model='deepseek-r1:8b', messages=[{'role': 'user', 'content': user_prompt}])
        deepseek_response = response.message.content.strip()

        message = await message_service.create_message(
            chat_id=chat_id,
            text=user_prompt,
            response=deepseek_response
        )

        return message

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating Deepseek response: {str(e)}")
    
    
    

# Endpoint for getting a structured response from Deepseek model
@router.post("/deepseek_response-stru", tags=["LLM"], summary="Send user prompt to Deepseek model and save response")
async def get_deepseek_response(
    chat_id: str = Query(..., description="Chat session ID"),
    user_prompt: str = Query(..., description="User input prompt for Deepseek model"),
    message_service: MessageService = Depends(get_message_service),
):
    try:
        # Send the user prompt to Deepseek model via the ollama API
        response = chat(model='deepseek-r1:8b', messages=[{'role': 'user', 'content': user_prompt}])

        # Extract content from the response
        deepseek_response = response.message.content.strip()

        # Structure the response in a more readable format
        structured_response = {
            "chat_id": chat_id,
            "user_prompt": user_prompt,
            "deepseek_response": deepseek_response,
            "model": "deepseek-r1:8b",
            "response_length": len(deepseek_response),
        }

        # Create and save the message with the user prompt and Deepseek response
        message = await message_service.create_message(
            chat_id=chat_id,
            text=user_prompt,
            response=deepseek_response
        )

        # Return the structured response
        return structured_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating Deepseek response: {str(e)}")
    
    
# Endpoint for feeding chunks to Deepseek LLM and getting response
@router.post("/test_chunks_deepseek", tags=["LLM"], summary="Perform hybrid search and send result to Deepseek")
async def test_chunks_deepseek(
    chat_id: str = Query(..., description="Chat session ID"),
    user_prompt: str = Query(..., description="User input prompt for Deepseek"),
    message_service: MessageService = Depends(get_message_service),
):
    try:
        # Perform hybrid search to retrieve relevant chunks using the hybrid_search method
        faiss_results = message_service.hybrid_search(query=user_prompt, chat_id=chat_id, top_k=5, min_words=5)

        # Prepare the chunks for the prompt (you can adjust how many chunks you want to include)
        combined_chunks = " ".join([chunk["text"] for chunk in faiss_results])  # Combine the top chunks
        
        print("Combined Chunks:", combined_chunks)

        # Concatenate the user prompt with the relevant chunks
        full_prompt = combined_chunks + "\n" + user_prompt

        # Send the full prompt to Deepseek via the ollama API
        response = chat(model='deepseek-r1:8b', messages=[{'role': 'user', 'content': full_prompt}])

        # Extract the response text from Deepseek
        deepseek_response = response.message.content.strip()
        
        # Remove thinking text if any
        if deepseek_response.startswith("<think>"):
            deepseek_response = deepseek_response.split("</think>")[-1].strip()

        # Create and save the message with the user prompt and Deepseek response
        message = await message_service.create_message(
            chat_id=chat_id,
            text=user_prompt,
            response=deepseek_response
        )

        # Return the saved message with the response
        return message

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chunks and generating Deepseek response: {str(e)}")


# @router.post("/generate-quiz", tags=["LLM"], summary="Generate quiz from text")
# async def generate_quiz(user_prompt: str = Query(..., description="Text input to generate quiz from")):
#     prompt = f"""
#     Given the following text, create exactly 5 multiple-choice quiz questions.

#     Strictly return ONLY JSON. Do NOT include any explanation, comments, or additional text outside the JSON.

#     JSON format example:
#     {{
#       "questions": [
#         {{
#           "description": "Question text?",
#           "options": ["Option A", "Option B", "Option C", "Option D"],
#           "answer": "Correct Option"
#         }}
#       ]
#     }}

#     Text to use:
#     {user_prompt}

#     JSON:
#     """

#     try:
#         output = llm(
#             prompt,
#             max_tokens=1000,
#             temperature=0.5,
#             stop=["\n\n"]
#         )
#         response_text = output["choices"][0]["text"].strip()

#         json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
#         if not json_match:
#             raise ValueError("No JSON found in the LLM response.")

#         json_content = json_match.group(0)

#         quiz_json = json.loads(json_content)
#         return quiz_json

#     except json.JSONDecodeError as e:
#         raise HTTPException(status_code=500, detail=f"Invalid JSON returned by LLM. Error: {str(e)}")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
