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

@router.post("/create-chat", tags=["chat"], summary="Upload a PDF and create chat")
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

@router.get("/all-chats",tags=["chat"], summary="Get all chats")
async def get_all_chats(chat_service: ChatService = Depends(get_chat_service)):
    chats = await chat_service.get_all_chats()
    return {"chats": chats}

@router.get("/user-chats/{user_id}",tags=["User"], summary="Get chats by user ID")
async def get_user_chats(user_id: str, chat_service: ChatService = Depends(get_chat_service)):
    chats = await chat_service.get_chats_by_user_id(user_id)
    return {"chats": chats}

@router.put("/edit-chat/{chat_id}", tags=["chat"], summary="Edit chat details")
async def edit_chat(chat_id: str, title: Optional[str] = None, chat_service: ChatService = Depends(get_chat_service)):
    chat = await chat_service.get_chat_by_id(chat_id)
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Update the fields that are provided
    updated_chat = {}
    if title:
        updated_chat["title"] = title

    if updated_chat:
        await chat_service.update_chat(chat_id, updated_chat)
        return {"message": "Chat updated successfully", "chat_id": chat_id}
    else:
        raise HTTPException(status_code=400, detail="No fields to update")


@router.delete("/delete-chat/{chat_id}", tags=["chat"], summary="Delete a chat")
async def delete_chat(chat_id: str, chat_service: ChatService = Depends(get_chat_service)):
    chat = await chat_service.get_chat_by_id(chat_id)
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    await chat_service.delete_chat(chat_id)
    return {"message": "Chat deleted successfully", "chat_id": chat_id}