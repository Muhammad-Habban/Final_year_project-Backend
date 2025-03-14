from pydantic import BaseModel
from datetime import datetime
from uuid import uuid4

class Message(BaseModel):
    id: str
    chat_id: str
    user_id: str
    text: str
    response: str
    timestamp: str

    @classmethod
    def create(cls, chat_id: str, user_id: str, text: str, response: str):
        return cls(
            id=str(uuid4()),
            chat_id=chat_id,
            user_id=user_id,
            text=text,
            response=response,
            timestamp=datetime.utcnow().isoformat()
        )