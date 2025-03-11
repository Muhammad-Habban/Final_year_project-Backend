from typing import Union, Any
from datetime import datetime
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from middleware import (
    ALGORITHM,
    JWT_SECRET_KEY
)

import jwt
from pydantic import ValidationError, BaseModel
from models.user import SystemUser
from database import get_database
from services.user_service import UserService
from repositories.user_repository import UserRepository

class TokenPayload(BaseModel):
    sub: str = None
    exp: int = None

def get_user_service(db=Depends(get_database)):
    user_repo = UserRepository(db['users'])
    return UserService(user_repo)

reuseable_oauth = OAuth2PasswordBearer(
    tokenUrl="/login",
    scheme_name="JWT"
)


async def get_current_user(token: str = Depends(reuseable_oauth), user_service: UserService = Depends(get_user_service)) -> SystemUser:
    try:
        payload = jwt.decode(
            token, JWT_SECRET_KEY, algorithms=[ALGORITHM]
        )
        token_data = TokenPayload(**payload)
        
        if datetime.fromtimestamp(token_data.exp) < datetime.now():
            raise HTTPException(
                status_code = status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except(jwt.JWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user: Union[dict[str, Any], None] = await user_service.find_user_by_email(token_data.sub)
    
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Could not find user",
        )
    
    return SystemUser(**user)