from pydantic import BaseModel
from typing import List

class Chat(BaseModel):
    chat_id: str
    user_id: str
    path: str
    title: str
    created_at: str
    updated_at: str
