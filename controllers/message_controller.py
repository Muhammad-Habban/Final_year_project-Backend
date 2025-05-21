# Standard library imports
import glob
from fastapi import APIRouter, Request, Query, Depends, HTTPException, File, UploadFile
import subprocess
import uuid
import warnings
import os
import re
import json
import tempfile
# Third-party library imports
import requests
import httpx
import numpy as np
import torch
from dotenv import load_dotenv

# Google & OpenAI imports
from googleapiclient.discovery import build
from google import generativeai as genai
import openai

# LangChain imports
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Transformers for NLP pipelines
from transformers import pipeline

# Ollama for local model chat
from ollama import chat

# Custom module imports
from repositories.message_repository import MessageRepository
from services.message_service import MessageService
from database import get_database
from models.message import Message

# Llama-cpp-python for local LLM
from llama_cpp import Llama

# Load environment variables
load_dotenv()

# Configure API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CSE_ID = os.getenv("GOOGLE_CSE_ID")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configure Google AI if API key is available
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Initialize FastAPI router
router = APIRouter()

# Initialize Gemini LLM with configuration
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.5,
    google_api_key=GOOGLE_API_KEY
)

# Define Gemini prompt template
gemini_prompt = PromptTemplate(
    input_variables=["user_prompt"],
    template="Question: {user_prompt}\nAnswer:"
)

# Create Gemini chain
gemini_chain = LLMChain(llm=gemini_llm, prompt=gemini_prompt)

# Initialize Llama model
model_path = os.getenv("LLAMA_MODEL_PATH", "llama-2-7b-chat.gguf")
llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_batch=512,
    use_mmap=True,
    verbose=False
)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def summarize_text(text: str, max_length: int = 130, min_length: int = 30) -> str:
    """
    Summarize the input text using the BART summarization pipeline.

    Args:
        text (str): The text to summarize.
        max_length (int): Maximum length of the summary.
        min_length (int): Minimum length of the summary.

    Returns:
        str: The summarized text.
    """
    result = summarizer(text, max_length=max_length,
                        min_length=min_length, do_sample=False)

    return result[0]['summary_text']

# Dependency injection for message service


def get_message_service(db=Depends(get_database)):
    return MessageService(MessageRepository(db['messages']))


# Enhanced response prompt template for teaching assistant role
enhanced_response_prompt = PromptTemplate.from_template("""
You are a helpful, patient teaching assistant for students.

Below is some context from a book or notes. Your job is to read it carefully, understand the key ideas, and respond to the student's question in a way that helps them learn.

---

 **Context:**  
{context}

 **Student's Question:**  
{question}

---

 Your Task:

1. **Explain the concept in simple, clear, and friendly language**  
2. **Include 1-2 real-world or relatable examples**  
3. **Ask 1-2 follow-up questions to test the student's understanding or provoke deeper thinking**

---

**Response Format:**

###  Explanation
(A clear and easy explanation of the concept using simple words.)

###  Example(s)
(At least one example, analogy, or case to reinforce the idea.)

### Follow-Up Questions
- Question 1  
- (Optional) Question 2

Make your tone friendly, supportive, and student-focused.
""")

# Quiz generation prompt template
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

# Create quiz chain
quiz_chain = quiz_prompt | llm


@router.get("/messages/{chat_id}", tags=["messages"], summary="Get messages by chat ID", response_model=list[Message])
async def get_messages_by_chat_id(
    chat_id: str,
    message_service: MessageService = Depends(get_message_service)
):
    """Retrieve all messages for a given chat ID."""
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
    """Generate a response using OpenAI's GPT-4 model."""
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


@router.post("/enhanced_response_open_ai", tags=["LLM"], summary="Retrieve enhanced response from GPT-4")
async def test_chunks(
    chat_id: str = Query(..., description="Chat session ID"),
    user_prompt: str = Query(..., description="User input prompt for LLM"),
    message_service: MessageService = Depends(get_message_service),
):
    """Generate an enhanced response using GPT-4 with context from previous messages."""
    try:
        # Retrieve relevant context from previous messages
        faiss_results = message_service.hybrid_search(
            query=user_prompt, chat_id=chat_id, top_k=5)
        combined_chunks = " ".join([chunk["text"] for chunk in faiss_results])

        # Format prompt with context
        full_prompt = enhanced_response_prompt.format(
            context=combined_chunks,
            question=user_prompt
        )

        # Generate response
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt},
            ],
            max_tokens=4096,
            temperature=0.7
        )
        llm_response = response['choices'][0]['message']['content'].strip()

        # Save message
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
    """Generate a response using the Deepseek model."""
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
    """Generate an enhanced response using Deepseek with context from previous messages."""
    try:
        # Retrieve relevant context
        faiss_results = message_service.hybrid_search(
            query=user_prompt, chat_id=chat_id, top_k=5)
        combined_chunks = " ".join([chunk["text"] for chunk in faiss_results])

        # Format prompt with context
        full_prompt = enhanced_response_prompt.format(
            context=combined_chunks,
            question=user_prompt
        )

        # Generate response
        response = chat(model='deepseek-r1:8b',
                        messages=[{'role': 'user', 'content': full_prompt}])
        deepseek_response = response.message.content.strip()

        # Clean response if needed
        if deepseek_response.startswith("<think>"):
            deepseek_response = deepseek_response.split("</think>")[-1].strip()

        # Save message
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
    """Generate a detailed response using the quantized Deepseek model with context."""
    try:
        # Retrieve relevant context
        faiss_results = message_service.hybrid_search(
            query=user_prompt,
            chat_id=chat_id,
            top_k=5
        )
        if not faiss_results:
            raise HTTPException(
                status_code=404, detail="No relevant content found")

        # Format context
        context = "\n".join([
            f"*Source {i+1}:* {chunk['text']}\n"
            for i, chunk in enumerate(faiss_results)
            if chunk.get("text")
        ])

        # Format prompt
        prompt = enhanced_response_prompt.format(
            context=context,
            question=user_prompt
        )

        # Generate response
        output = llm(
            prompt,
            max_tokens=2500,
            temperature=0.5,
            top_p=0.85,
            stop=["## End of Response"],
            echo=False
        )

        # Process response
        response_text = output["choices"][0]["text"].strip()
        response_text = re.sub(r"\n{3,}", "\n\n", response_text)
        response_text = re.sub(r"(?<!\n)\n(?!\n)", " ", response_text)

        if not response_text:
            raise HTTPException(status_code=500, detail="No answer generated")

        # Save message
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


@router.post("/gemini_flash_response", tags=["LLM"], summary="Send user prompt to Gemini Flash and get response")
async def get_gemini_flash_response(
    chat_id: str = Query(..., description="Chat session ID"),
    user_prompt: str = Query(..., description="User input prompt for LLM"),
    message_service: MessageService = Depends(get_message_service),
):
    """Generate a response using Google's Gemini Flash model."""
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


@router.post("/enhanced_gemini_response", tags=["LLM"], summary="Retrieve enhanced response from Gemini Flash")
async def enhanced_gemini_response(
    chat_id: str = Query(..., description="Chat session ID"),
    user_prompt: str = Query(..., description="User input prompt for Gemini"),
    message_service: MessageService = Depends(get_message_service),
):
    """Generate an enhanced response using Gemini Flash with context from previous messages."""
    try:
        # Retrieve relevant context
        faiss_results = message_service.hybrid_search(
            query=user_prompt, chat_id=chat_id, top_k=5)
        combined_chunks = " ".join([chunk["text"] for chunk in faiss_results])

        # Format prompt with context
        full_prompt = enhanced_response_prompt.format(
            context=combined_chunks,
            question=user_prompt
        )

        # Generate response
        response = gemini_chain.run(full_prompt)
        llm_response = response.strip()

        # Save message
        message = await message_service.create_message(
            chat_id=chat_id,
            text=user_prompt,
            response=llm_response
        )
        return message
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chunks and generating Gemini Flash response: {str(e)}"
        )


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
    """Search for web results and associated images using Google Custom Search."""
    if not GOOGLE_API_KEY or not CSE_ID:
        raise HTTPException(
            status_code=500, detail="API key or CSE ID not set.")
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        web_results = service.cse().list(q=query, cx=CSE_ID, num=3).execute()
        web_items = web_results.get("items", [])
        final_results = {}

        # Process each web result and get associated images
        for item in web_items:
            title = item.get("title")
            link = item.get("link")
            snippet = item.get("snippet")

            # Search for images related to the web result
            image_search = service.cse().list(
                q=title,
                cx=CSE_ID,
                searchType="image",
                num=2
            ).execute()

            # Process image results
            image_items = image_search.get("items", [])
            images = []
            for img in image_items:
                images.append({
                    "title": img.get("title"),
                    "image_url": img.get("link"),
                    "context_link": img.get("image", {}).get("contextLink")
                })

            # Store results
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


def cleanup_generated_files(base_filename: str, image_filenames: list[str]):
    # Delete images
    for img_file in image_filenames:
        if os.path.exists(img_file):
            os.remove(img_file)

    # Delete .tex source file
    tex_file = f"{base_filename}.tex"
    if os.path.exists(tex_file):
        os.remove(tex_file)

    # Delete auxiliary files created by pdflatex
    extensions = ["aux", "log", "out", "toc", "nav", "snm"]
    for ext in extensions:
        file_path = f"{base_filename}.{ext}"
        if os.path.exists(file_path):
            os.remove(file_path)


def extract_latex_code(raw_response):
    lines = raw_response.split("\n")
    in_code_block = False
    cleaned_lines = []
    for line in lines:
        if line.strip().startswith("```latex"):
            in_code_block = True
            continue
        if line.strip() == "```":
            in_code_block = False
            continue
        if in_code_block:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


@router.post("/generate-beamer-slide", tags=["LLM"])
async def generate_beamer_slide(
    message_id: str = Query(...,
                            description="Message ID to generate beamer slide for"),
    request: Request = None,
    message_service: MessageService = Depends(get_message_service)
):
    """Generate a LaTeX beamer slide from a message with associated images."""
    try:
        # Get message content
        message = await message_service.get_message_by_id(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        message = dict(message)
        text, response = message.get("text"), message.get("response")
        if not text or not response:
            raise HTTPException(
                status_code=400, detail="Message missing text or response")

        # Search for relevant images
        base_url = str(
            request.base_url) if request else "http://localhost:8000/"
        async with httpx.AsyncClient() as client:
            img_result = await client.post(f"{base_url}search-with-images", params={"query": text})
        if img_result.status_code != 200:
            raise HTTPException(status_code=500, detail="Image search failed")
        images_data = img_result.json()

        # Generate LaTeX code with image downloads
        download_cmds = []
        include_lines = []
        image_filenames = []
        reference_links = []
        img_counter = 1

        for title, result in images_data.get("results", {}).items():
            for img in result.get("images", []):
                img_url = img.get("image_url", "")
                ext = img_url.split(".")[-1].split("?")[0]
                if ext == "jpg" or ext == "jpeg":
                    filename = f"image{img_counter}.{ext}"
                    download_cmds.append(f"curl -o {filename} {img_url}")
                    include_lines.append(
                        f"\\includegraphics[width=0.45\\textwidth]{{{filename}}}\\\\\n"
                        f"\\textit{{{img.get('title', '')}}}\\\\\n\\vspace{{1em}}"
                    )
                    image_filenames.append(filename)
                    img_counter += 1
            link = result.get("link")
            if link:
                reference_links.append(link)

        print(reference_links)

        write18_block = "\n".join(
            [f"\\immediate\\write18{{{cmd}}}" for cmd in download_cmds])
        image_block = "\n".join(include_lines)

        openai_prompt = f"""
GENERATE CORRECT SYNTAX FOR A LATEX BEAMER DOCUMENT THAT:

- Includes the following text in a well-formatted slide(s):
SLIDE 1 AND 2:
{response}
EXPAND ON THIS RESPONSE IF NECESSARY, CREATE BEAUTIFULL BULLET POINTS AND EXPLAIN IT
SLIDE 3 AND 4:
- Adds these images at appropriate places, with captions, using the local filenames below:
{image_block}
DO ADD A CAPTION TO EACH IMAGE IF YOU CAN, AND ADD MORE SLIDES IF NEEDED TO COVER ALL THE IMAGES
SLIDE 5:
ADD THE FOLLOWING REFERAL LINK
{reference_links}

- Starts the LaTeX code with these commands to download images dynamically:
{write18_block}

Return ONLY the complete LaTeX code, starting with \\documentclass and ending with \\end{{document}}.
IMPORTANT: ONLY GENERATE LATEX CODE IN RESPONSE, DO NOT SAY ANYTHING ELSE

IMPORTANT: wrap every \\includegraphics in a \\IfFileExists check
"""

        # 5. Call OpenAI ChatCompletion
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a LaTeX Beamer code generator."},
                {"role": "user", "content": openai_prompt}
            ],
            max_tokens=4096,
            temperature=0.7,
        )
        latex_code = completion.choices[0].message.content.strip()
        latex_code = extract_latex_code(latex_code)
        # Write and compile LaTeX
        file_id = uuid.uuid4().hex
        tex_file = f"slide_{file_id}.tex"
        base_filename = f"slide_{file_id}"
        with open(tex_file, "w") as f:
            f.write(latex_code)

        subprocess.run(["pdflatex", "--shell-escape", tex_file], check=True)

        # Return PDF path
        cleanup_generated_files(base_filename=base_filename,
                                image_filenames=image_filenames)
        pdf_path = tex_file.replace(".tex", ".pdf")
        if not os.path.exists(pdf_path):
            raise HTTPException(
                status_code=500, detail="PDF generation failed")

        return {
            "status": "success",
            "pdf_path": pdf_path
        }

    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500, detail=f"LaTeX compile error: {e}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unhandled error: {str(e)}")


def search_youtube_videos(query: str, api_key: str, max_results: int = 5):
    """Search for YouTube videos using the YouTube Data API."""
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


@router.get("/youtube-search", tags=["Web Search"], summary="Search YouTube videos")
def youtube_search(q: str = Query(..., description="Search query"), max_results: int = 5):
    """Search for YouTube videos and return formatted results."""
    results = search_youtube_videos(q, YOUTUBE_API_KEY, max_results)
    output = []
    for item in results.get("items", []):
        output.append({
            "title": item["snippet"]["title"],
            "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
        })
    return {"results": output}
