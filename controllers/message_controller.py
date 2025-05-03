import openai
from fastapi import APIRouter, HTTPException, Query, Depends, File, UploadFile
from repositories.message_repository import MessageRepository
from services.message_service import MessageService
from database import get_database
from models.message import Message
import os
import numpy as np
from transformers import pipeline
from ollama import chat
import json
import re
from llama_cpp import Llama
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
import soundfile as sf
import tempfile
from pydub import AudioSegment
import torch
from googleapiclient.discovery import build
from dotenv import load_dotenv
load_dotenv()
# Load OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
API_KEY = os.getenv("GOOGLE_API_KEY")
CSE_ID = os.getenv("GOOGLE_CSE_ID")
router = APIRouter()

# Load GGUF quantized model

model_path = "E:\\deepseek-llm-7b-chat.Q4_K_M.gguf"
llm = LlamaCpp(
    model_path=model_path,
    n_ctx=4096,
    max_tokens=1000,
    temperature=0.7
)

device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=device)
SAMPLE_RATE = 16000


#____________________________________________________________________________________    
def get_message_service(db=Depends(get_database)):
    return MessageService(MessageRepository(db['messages']))


#____________________________________________________________________________________    
# Convert MP3 to WAV (needed for the whisper model)
def mp3_to_wav(mp3_file):
    audio = AudioSegment.from_mp3(mp3_file)
    audio = audio.set_channels(1)  # Convert to mono (1 channel)
    wav_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    audio.export(wav_file.name, format="wav")
    return wav_file.name


#____________________________________________________________________________________    
# Function to process audio and return transcription
def transcribe_audio(audio_file):
    wav_path = mp3_to_wav(audio_file)
    audio, samplerate = sf.read(wav_path)
    audio = np.squeeze(audio)

    transcription = whisper_pipeline({"sampling_rate": SAMPLE_RATE, "raw": audio},generate_kwargs={"language": "en", "return_timestamps": True})
    return transcription


#____________________________________________________________________________________    
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


#____________________________________________________________________________________    
@router.post("/open_ai_response", tags=["LLM"], summary="Send user prompt to GPT-4 and get response")
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



#____________________________________________________________________________________    
# Endpoint for feeding chunks to LLM and getting response
@router.post("/enhanced_response_open_ai", tags=["LLM"], summary="Retrieve enhanced response from GPT-4")
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



#____________________________________________________________________________________    
# Getting response from DeepSeek
@router.post("/deepseek_response", tags=["LLM"], summary="Send user prompt to Deepseek model and get response")
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
 
    
    
#____________________________________________________________________________________      
# Endpoint for feeding chunks to Deepseek LLM and getting response
@router.post("/enhanced_response_deepseek", tags=["LLM"], summary="Retrieve enhanced response from Deepseek")
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



#____________________________________________________________________________________    
@router.post("/deepseek_q_response", tags=["LLM"], summary="Get response from quantized Deepseek model")
async def simple_answer(
    chat_id: str = Query(..., description="Chat session ID"),
    user_prompt: str = Query(..., description="User input prompt for the LLM"),
    message_service: MessageService = Depends(get_message_service),
):
    try:
        # Step 1: Retrieve relevant chunks using hybrid search
        faiss_results = message_service.hybrid_search(
            query=user_prompt,
            chat_id=chat_id,
            top_k=5
        )
        combined_chunks = " ".join(chunk["text"] for chunk in faiss_results)

        # Step 2: Build a structured prompt with guidance and context
        refinement_prompt = f"""
You are a highly knowledgeable physics expert. Your task is to generate a clear, detailed, and well-structured response to my question, ensuring a strong conceptual understanding and a professional, high-quality impression.

### My Question:
{user_prompt}

### Instructions:
- Carefully analyze the provided reference text and use only **the most relevant parts** that closely relate to the question.
- **Ignore any chunks** that are unrelated or do not contribute meaningfully to the response.
- Craft a well-structured response that explains the core concept in **simple, precise, and engaging language**.
- Use **real-world examples** to illustrate the topic effectively.
- Identify and describe any **types, categories, or variations** relevant to the concept.
- Incorporate **relevant formulas and equations**, ensuring each variable is clearly defined.
- Provide **step-by-step derivations** where necessary to enhance clarity.
- Include **key insights, interesting facts, or historical context** to make the explanation more engaging.
- Ensure the response is **comprehensive, professional, and insightful** to leave a strong impression.
- Conclude with a set of **engaging follow-up questions** based on the explanation, ensuring that each question aligns with a concept already covered in the text.

### Reference Text:
{combined_chunks}

### Final Output:
1. A **refined, structured response** incorporating deep explanations, examples, formulas, and derivations.
2. A set of **relevant, thought-provoking tidbit questions** for the student to test their understanding, ensuring each question aligns with a concept already explained in the response.
"""

        print(refinement_prompt)
        
        # Step 3: Generate response using the LLM
        response_text = llm.invoke(refinement_prompt).strip()
        print("______________________________________________")
        print(response_text)
        # Step 4: Save and return the response
        message = await message_service.create_message(
            chat_id=chat_id,
            text=user_prompt,
            response=response_text
        )
        return message

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating Deepseek response: {str(e)}"
        )
        

#____________________________________________________________________________________         
@router.post("/transcribe", tags=["LLM"], summary="Transcribe audio file")
async def transcribe( chat_id: str = Query(..., description="Chat session ID"), file: UploadFile = File(...)):
    try:
        # Transcribe the audio file
        transcription = transcribe_audio(file.file)
        
        return {"chat_id": chat_id, "transcription": transcription}

    except Exception as e:
        return {"error": str(e)}

    
    
#____________________________________________________________________________________    
@router.post("/voice-message", tags=["LLM"], summary="Input a text file")
async def transcribe( chat_id: str = Query(..., description="Chat session ID"), file: UploadFile = File(...)):
    try:
        # Transcribe the audio file
        transcription = transcribe_audio(file.file)
        result = await test_chunks(chat_id=chat_id, user_prompt=transcription)
        return {"chat_id": chat_id, "transcription": transcription, "result": result}

    except Exception as e:
        return {"error": str(e)}



#____________________________________________________________________________________    
quiz_prompt = PromptTemplate.from_template("""
Given the following text, create exactly 5 multiple-choice quiz questions.

Strictly return ONLY JSON. Do NOT include any explanation, comments, or additional text outside the JSON.

JSON format:
{{
  "questions": [
    {{
      "description": "Question text?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "answer": "Correct Option"
    }}
  ]
}}

Text to use:
{input_text}

JSON:
""")
quiz_chain = quiz_prompt | llm
@router.post("/generate-quiz", tags=["LLM"], summary="Generate quiz from text")
async def generate_quiz(user_prompt: str = Query(..., description="Text input to generate quiz from")):
    try:
        response = quiz_chain.invoke({"input_text": user_prompt})
        
        # Just to be safe, extract JSON from potentially noisy output
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in the LLM response.")

        quiz_json = json.loads(json_match.group(0))
        return quiz_json

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON returned by LLM. Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    
  
  
  
#____________________________________________________________________________________        
@router.get("/search-with-images", tags=["Web Search"], summary="3 web results + 2 images each")
def search_with_images(query: str = Query(..., description="Search query")):
    if not API_KEY or not CSE_ID:
        raise HTTPException(status_code=500, detail="API key or CSE ID not set.")
    try:
        service = build("customsearch", "v1", developerKey=API_KEY)

        # Step 1: Web Search
        web_results = service.cse().list(q=query, cx=CSE_ID, num=3).execute()
        web_items = web_results.get("items", [])

        final_results = []

        for item in web_items:
            title = item.get("title")
            link = item.get("link")
            snippet = item.get("snippet")

            # Step 2: Image Search using the web page title or URL
            image_search = service.cse().list(
                q=title,  # or use `link` if you prefer
                cx=CSE_ID,
                searchType="image",
                num=2
            ).execute()
            image_items = image_search.get("items", [])

            images = []
            for img in image_items:
                images.append({
                    "title": img.get("title"),
                    "image_url": img.get("link"),
                    "context_link": img.get("image", {}).get("contextLink")
                })

            final_results.append({
                "title": title,
                "link": link,
                "snippet": snippet,
                "images": images
            })

        return {
            "query": query,
            "results": final_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")