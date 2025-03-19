from pydantic import BaseModel

class RequestBody(BaseModel):
    chat_id: str
    user_id: str
    user_prompt: str