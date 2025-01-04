from pydantic import BaseModel
from typing import List

class Chat(BaseModel):
    chat_id: str
    messages: List[dict]  # This could be more detailed based on your requirements
