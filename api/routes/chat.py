from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


@router.post("/")
async def create_chat():
    pass