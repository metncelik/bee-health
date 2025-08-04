from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from services.chat import chat_service

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)

class CreateChatRequest(BaseModel):
    prediction_id: int

class AddMessageRequest(BaseModel):
    content: str

@router.post("/")
async def create_chat(request: CreateChatRequest):
    """Create a new chat session for a prediction"""
    return chat_service.create_chat(request.prediction_id)

@router.post("/{chat_id}/messages")
async def add_message(chat_id: int, request: AddMessageRequest):
    """Add a message to a chat"""
    return chat_service.add_message(chat_id, request.content)

@router.get("/{chat_id}/messages")
async def get_chat_messages(chat_id: int):
    """Get all messages in a chat"""
    return chat_service.get_chat_messages(chat_id)

@router.get("/{chat_id}")
async def get_chat_details(chat_id: int):
    """Get chat details with prediction context"""
    return chat_service.get_chat_details(chat_id)