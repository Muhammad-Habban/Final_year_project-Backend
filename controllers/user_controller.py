from services.user_service import UserService
from repositories.user_repository import UserRepository
from database import get_database
from fastapi import APIRouter, status, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm
from models.user import UserOut, UserAuth, SystemUser
from deps import get_current_user
from middleware import (
    get_hashed_password,
    create_access_token,
    create_refresh_token,
    verify_password
)
from uuid import uuid4
from pydantic import BaseModel

class TokenSchema(BaseModel):
    access_token: str
    refresh_token: str
    
    
class TokenPayload(BaseModel):
    sub: str = None
    exp: int = None

def get_user_service(db=Depends(get_database)):
    user_repo = UserRepository(db['users'])
    return UserService(user_repo)

router = APIRouter()


# app = FastAPI()

@router.post('/signup', summary="Create new user", response_model=UserOut)
async def create_user(data: UserAuth,  user_service: UserService = Depends(get_user_service)):
    user = await user_service.find_user_by_email(email=data.email)
    print("User : ")
    print(user)
    if user is not None:
            raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exist"
        )
    data.password = get_hashed_password(data.password)
    returned_user = await user_service.create_user(data.dict())
    print("returned_user : ")
    print(returned_user)
    fun_ret_user = {
        'email': returned_user['email'],
        'id': returned_user['id']
    }
    return fun_ret_user


@router.post('/login', summary="Create access and refresh tokens for user", response_model=TokenSchema)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), user_service: UserService = Depends(get_user_service)):
    user = await user_service.find_user_by_email(form_data.username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect email or password"
        )

    hashed_pass = user['password']
    if not verify_password(form_data.password, hashed_pass):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect email or password"
        )
    
    return {
        "access_token": create_access_token(user['email']),
        "refresh_token": create_refresh_token(user['email']),
    }

@router.get('/me', summary='Get details of currently logged in user', response_model=UserOut)
async def get_me(user: SystemUser = Depends(get_current_user)):
    return user