from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from typing import Optional
from services.chat_service import ChatService
from repositories.chat_repository import ChatRepository
from database import get_database
import aiofiles


def get_chat_service(db=Depends(get_database)):
    chat_repository = ChatRepository(db.get_collection('chats'))
    return ChatService(chat_repository=chat_repository)

router = APIRouter()

@router.post("/create-chat", summary="Upload a PDF and create chat")
async def upload_pdf(file: UploadFile = File(...), user_id: str = Form(...), chat_service: ChatService = Depends(get_chat_service)):
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=415, detail="Unsupported file type")
    
    # Create chat object
    chat = await chat_service.create_chat(user_id, file.filename)
    
    # Process PDF file
    try:
        path = f"uploads/{chat['chat_id']}.pdf"
        async with aiofiles.open(path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        # Update chat object with file path
        await chat_service.update_chat_path(chat['chat_id'], path)

        # Call create_chunks function
        await chat_service.create_chunks(chat['chat_id'], path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"message": "File uploaded successfully", "chat_id": chat['chat_id']}
