from pydantic import BaseModel, Field

class UserAuth(BaseModel):
    email: str = Field(..., description="user email")
    password: str = Field(..., min_length=5, max_length=24, description="user password")
    

class UserOut(BaseModel):
    id: str
    email: str


class SystemUser(UserOut):
    password: str