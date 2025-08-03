from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime
from fastapi import UploadFile, File
from services.predict import predict_service

router = APIRouter(
    prefix="/predict",
    tags=["predict"],
)

@router.post("/")
async def predict(image: UploadFile = File(...)):
    return predict_service.predict(image)