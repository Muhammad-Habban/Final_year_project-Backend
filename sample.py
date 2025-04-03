from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.output_parsers import PydanticOutputParser

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Ollama configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
# MODEL_NAME = "deepseek-r1:8b-llama-distill-q8_0"
# MODEL_NAME = "deepseek-r1:1.5b-qwen-distill-q8_0"
MODEL_NAME = "deepseek-r1:1.5b-qwen-distill-fp16"
# MODEL_NAME= "deepseek-r1:7b-qwen-distill-q8_0"

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        # Prepare the request data for Ollama
        ollama_data = {
            "model": MODEL_NAME,
            "prompt": request.question,
            "stream": False
        }

        # Send request to Ollama
        response = requests.post(OLLAMA_API_URL, json=ollama_data)
        response.raise_for_status()
        print(f"-------------- {response}---------------")
        # Extract the response content
        response_data = response.json()
        return {"answer": response_data.get("response", "No answer generated")}
    
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Ollama service is not running. Please ensure Ollama is installed and running."
        )
    except requests.exceptions.HTTPError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error from Ollama API: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )


# Define Pydantic model for output
class OrderInfo(BaseModel):
    customer_name: Optional[str] = None
    issue_type: str = "Order Tracking"
    order_id: Optional[str] = None
    urgency: str = "Medium"
    requested_action: str = "Track Order"

# Create output parser
parser = PydanticOutputParser(pydantic_object=OrderInfo)

# Define prompt template
template = """Extract the following information from the customer service context:
- customer_name: Full name of the customer (if mentioned)
- issue_type: Only use one of these: [Order Tracking, Return Request, Delivery Issue, Product Complaint]
- order_id: Any order identification number found
- urgency: Assess urgency as Low, Medium or High
- requested_action: What the customer is asking to do

Return JSON with the keys: customer_name, issue_type, order_id, urgency, requested_action.

Context: {context}

{format_instructions}"""

prompt = ChatPromptTemplate.from_template(template)

# Initialize Ollama
llm = Ollama(model=MODEL_NAME)
chain = prompt | llm | parser

# Input/Output models
class ContextRequest(BaseModel):
    context: str

class OrderInfoResponse(OrderInfo):
    pass

@app.post("/extract-order-info", response_model=OrderInfoResponse)
async def extract_order_info(request: ContextRequest):
    try:
        # Invoke the LangChain pipeline
        result = chain.invoke({
            "context": request.context,
            "format_instructions": parser.get_format_instructions()
        })
        return result.dict()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )