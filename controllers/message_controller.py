# Standard
import os
import re
import json
import tempfile

#  3rd Party Libraries
import requests
import httpx
import numpy as np
import torch  

from fastapi import (
    APIRouter, HTTPException, Query, Depends, File, UploadFile, Request
)
from dotenv import load_dotenv

# Google & OpenAI
from googleapiclient.discovery import build
from google import generativeai as genai
import openai

# ✅ LangChain
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ Transformers (used for NLP pipelines like summarization, classification, etc.)
from transformers import pipeline

# ✅ Ollama for local model chat
from ollama import chat

# ✅ Custom Modules
from repositories.message_repository import MessageRepository
from services.message_service import MessageService
from database import get_database
from models.message import Message

# llama-cpp-python
from llama_cpp import Llama


load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CSE_ID = os.getenv("GOOGLE_CSE_ID")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

router = APIRouter()

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)

gemini_prompt = PromptTemplate(
    input_variables=["user_prompt"],
    template="Question: {user_prompt}\nAnswer:"
)

gemini_chain = LLMChain(llm=gemini_llm, prompt=gemini_prompt)

model_path = "E:/deepseek-llm-7b-chat.Q4_K_M.gguf"
llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_batch=512,
    use_mmap=True,
    verbose=False
)


def get_message_service(db=Depends(get_database)):
    return MessageService(MessageRepository(db['messages']))


@router.get("/messages/{chat_id}", tags=["messages"], summary="Get messages by chat ID", response_model=list[Message])
async def get_messages_by_chat_id(
    chat_id: str,
    message_service: MessageService = Depends(get_message_service)
):
    try:
        messages = await message_service.get_messages_by_chat_id(chat_id)
        return messages
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching messages: {str(e)}")


@router.post("/open_ai_response", tags=["LLM"], summary="Send user prompt to GPT-4 and get response")
async def get_response(
    chat_id: str = Query(..., description="Chat session ID"),
    user_prompt: str = Query(..., description="User input prompt for LLM"),
    message_service: MessageService = Depends(get_message_service),
):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        llm_response = response['choices'][0]['message']['content'].strip()
        message = await message_service.create_message(
            chat_id=chat_id,
            text=user_prompt,
            response=llm_response
        )
        return message
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {str(e)}")


@router.post("/gemini_flash_response", tags=["LLM"], summary="Send user prompt to Gemini Flash and get response")
async def get_gemini_flash_response(
    chat_id: str = Query(..., description="Chat session ID"),
    user_prompt: str = Query(..., description="User input prompt for LLM"),
    message_service: MessageService = Depends(get_message_service),
):
    try:
        response = gemini_chain.run(user_prompt)
        llm_response = response.strip()
        message = await message_service.create_message(
            chat_id=chat_id,
            text=user_prompt,
            response=llm_response
        )
        return message
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating Gemini Flash response: {str(e)}")


@router.post("/enhanced_response_open_ai", tags=["LLM"], summary="Retrieve enhanced response from GPT-4")
async def test_chunks(
    chat_id: str = Query(..., description="Chat session ID"),
    user_prompt: str = Query(..., description="User input prompt for LLM"),
    message_service: MessageService = Depends(get_message_service),
):
    try:
        faiss_results = message_service.hybrid_search(
            query=user_prompt, chat_id=chat_id, top_k=5)
        combined_chunks = " ".join([chunk["text"] for chunk in faiss_results])
        print("Combined Chunks:", combined_chunks)
        full_prompt = combined_chunks + "\n" + user_prompt
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt},
            ],
            max_tokens=1000,
            temperature=0.7
        )
        llm_response = response['choices'][0]['message']['content'].strip()
        message = await message_service.create_message(
            chat_id=chat_id,
            text=user_prompt,
            response=llm_response
        )
        return message
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing chunks and generating response: {str(e)}")


@router.post("/deepseek_response", tags=["LLM"], summary="Send user prompt to Deepseek model and get response")
async def get_deepseek_response(
    chat_id: str = Query(..., description="Chat session ID"),
    user_prompt: str = Query(...,
                             description="User input prompt for Deepseek model"),
    message_service: MessageService = Depends(get_message_service),
):
    try:
        response = chat(model='deepseek-r1:8b',
                        messages=[{'role': 'user', 'content': user_prompt}])
        deepseek_response = response.message.content.strip()
        message = await message_service.create_message(
            chat_id=chat_id,
            text=user_prompt,
            response=deepseek_response
        )
        return message
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating Deepseek response: {str(e)}")


@router.post("/enhanced_response_deepseek", tags=["LLM"], summary="Retrieve enhanced response from Deepseek")
async def test_chunks_deepseek(
    chat_id: str = Query(..., description="Chat session ID"),
    user_prompt: str = Query(...,
                             description="User input prompt for Deepseek"),
    message_service: MessageService = Depends(get_message_service),
):
    try:
        faiss_results = message_service.hybrid_search(
            query=user_prompt, chat_id=chat_id, top_k=5)
        combined_chunks = " ".join([chunk["text"] for chunk in faiss_results])
        print("Combined Chunks:", combined_chunks)
        full_prompt = combined_chunks + "\n" + user_prompt
        response = chat(model='deepseek-r1:8b',
                        messages=[{'role': 'user', 'content': full_prompt}])
        deepseek_response = response.message.content.strip()
        if deepseek_response.startswith("<think>"):
            deepseek_response = deepseek_response.split("</think>")[-1].strip()
        message = await message_service.create_message(
            chat_id=chat_id,
            text=user_prompt,
            response=deepseek_response
        )
        return message
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing chunks and generating Deepseek response: {str(e)}")


@router.post("/deepseek_q_response", tags=["LLM"], summary="Get detailed response from quantized Deepseek model")
async def get_detailed_answer(
    chat_id: str = Query(..., description="Chat session ID"),
    user_prompt: str = Query(..., description="User input prompt for the LLM"),
    message_service: MessageService = Depends(get_message_service),
):
    try:
        faiss_results = message_service.hybrid_search(
            query=user_prompt,
            chat_id=chat_id,
            top_k=5
        )
        if not faiss_results:
            raise HTTPException(
                status_code=404, detail="No relevant content found")
        context = "\n".join([
            f"*Source {i+1}:* {chunk['text']}\n"
            for i, chunk in enumerate(faiss_results)
            if chunk.get("text")
        ])
        prompt = f"""
                    *Enhanced Response Task*  

                    **Context:**  
                    {context}  

                    **User Prompt/Question:**  
                    {user_prompt}  

                    Using the information provided in the context, generate a **comprehensive, well-structured, and insightful explanation** of the topic or question. Ensure the response includes the following:

                    ---

                    ## 1. **Clear Explanation**  
                    - Provide a detailed and accurate explanation of the main topic or concept  
                    - Define key terms or ideas in simple, precise language  
                    - Address the user's question directly, with clarity and depth  

                    ---

                    ## 2. **Use of Contextual Information**  
                    - Integrate relevant facts, data, or references from the given context  
                    - Highlight connections between context and the topic  
                    - Avoid repeating the context verbatim—use it to **add insight**  

                    ---

                    ## 3. **Examples and Illustrations**  
                    - Include at least two real-world or relatable examples  
                    - Use analogies, case studies, or comparisons if helpful  
                    - Where applicable, describe diagrams, visual models, or scenarios  

                    ---

                    ## 4. **Fact-Based and Logical Reasoning**  
                    - Support explanations with logic, evidence, or citations where needed  
                    - Avoid vague claims—make statements grounded in knowledge or context  
                    - If applicable, provide numbers, dates, or references  

                    ---

                    ## 5. **Broader Relevance or Implications**  
                    - Explain why this topic matters or how it connects to larger themes  
                    - Mention applications, consequences, or cross-disciplinary relevance  
                    - Optionally, suggest further reading or questions for reflection  

                    ---

                    ### **Formatting Guidelines:**  
                    - Use **HTML** format  
                    - Section headers, bullet points, and clear structure  
                    - Highlight examples and key insights  
                    - Describe any diagrams or visuals in words if necessary  

                    """
        print("Prompt:", prompt)
        output = llm(
            prompt,
            max_tokens=2500,
            temperature=0.5,
            top_p=0.85,
            stop=["## End of Response"],
            echo=False
        )
        response_text = output["choices"][0]["text"].strip()
        response_text = re.sub(r"\n{3,}", "\n\n", response_text)
        response_text = re.sub(r"(?<!\n)\n(?!\n)", " ", response_text)
        if not response_text:
            raise HTTPException(status_code=500, detail="No answer generated")
        message = await message_service.create_message(
            chat_id=chat_id,
            text=user_prompt,
            response=response_text
        )
        return message
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating detailed response: {str(e)}"
        )

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
        formatted_prompt = quiz_prompt.format(input_text=user_prompt)
        output = llm(
            formatted_prompt,
            max_tokens=1000,
            temperature=0.7,
            stop=["```", "JSON"]
        )
        response = output["choices"][0]["text"]
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in the LLM response.")
        quiz_json = json.loads(json_match.group(0))
        return quiz_json
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500, detail=f"Invalid JSON returned by LLM. Error: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/search-with-images", tags=["Web Search"], summary="3 web results + 2 images each")
def search_with_images(query: str = Query(..., description="Search query")):
    if not GOOGLE_API_KEY or not CSE_ID:
        raise HTTPException(
            status_code=500, detail="API key or CSE ID not set.")
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        web_results = service.cse().list(q=query, cx=CSE_ID, num=3).execute()
        web_items = web_results.get("items", [])
        final_results = {}
        for item in web_items:
            title = item.get("title")
            link = item.get("link")
            snippet = item.get("snippet")
            image_search = service.cse().list(
                q=title,
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
            # Use title as key, fallback to a safe unique key if title is None
            key = title if title else f"result_{len(final_results)+1}"
            final_results[key] = {
                "link": link,
                "snippet": snippet,
                "images": images
            }
        return {
            "query": query,
            "results": final_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/generate-beamer-slide", tags=["LLM"], summary="Generate beamer slide code with images for a message")
async def generate_beamer_slide(
    message_id: str = Query(...,
                            description="Message ID to generate beamer slide for"),
    request: Request = None,
    message_service: MessageService = Depends(get_message_service),
):
    try:
        # 1. Get the message from the database
        message = await message_service.get_message_by_id(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        # Convert to dict if needed
        if not isinstance(message, dict):
            message = dict(message)
        text = message.get("text")
        response = message.get("response")
        if not text or not response:
            raise HTTPException(
                status_code=400, detail="Message missing text or response")

        # 2. Call the search-with-images API with the message text
        base_url = str(
            request.base_url) if request else "http://localhost:8000/"
        async with httpx.AsyncClient() as client:
            search_images_result = await client.post(f"{base_url}search-with-images", params={"query": text})
        if search_images_result.status_code != 200:
            raise HTTPException(status_code=500, detail="Image search failed")
        images_data = search_images_result.json()

        # 3. Prepare the prompt for Gemini
        images_section = ""
        for result in images_data.get("results", []):
            images_section += f"\nSection: {result.get('title', '')}\n"
            for img in result.get("images", []):
                images_section += f"- Image: {img.get('image_url', '')} (context: {img.get('context_link', '')})\n"

        beamer_prompt = f"""
You are a LaTeX Beamer slide generator. Given the following response and related images, generate a complete Beamer slide code.

- The slide should summarize the response.
- For each image, include it in the slide using its URL (use \\includegraphics[width=0.4\\textwidth]{{<image_url>}}).
- Add a caption for each image based on its context or title.
- Use a clear title and structure.
- The slide should be visually appealing and suitable for a presentation.

Response to present:
{response}

Images to include:{images_section}

Return only the LaTeX Beamer code, nothing else.
"""

        # 4. Get the beamer code from Gemini
        beamer_code = gemini_chain.run(beamer_prompt)
        return {"beamer_code": beamer_code.strip()}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating beamer slide: {str(e)}")

def search_youtube_videos(query: str, api_key: str, max_results: int = 5):
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "key": api_key
    }
    response = requests.get(url, params=params)
    return response.json()


@router.get("/youtube-search",tags=["Web Search"], summary="Search YouTube videos")
def youtube_search(q: str = Query(..., description="Search query"), max_results: int = 5):
    results = search_youtube_videos(q, YOUTUBE_API_KEY, max_results)
    output = []
    for item in results.get("items", []):
        output.append({
            "title": item["snippet"]["title"],
            "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
        })
    return {"results": output}