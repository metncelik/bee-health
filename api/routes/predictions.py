from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from fastapi import UploadFile, File
from services.predictions import predict_service

router = APIRouter(
    prefix="/predictions",
    tags=["predictions"],
)

@router.get("/")
async def get_predictions():
    """Get all predictions"""
    return predict_service.get_predictions()

@router.post("/")
async def predict(image: UploadFile = File(...)):
    """Predict bee hive health from uploaded image"""
    return await predict_service.predict(image)

@router.get("/{prediction_id}")
async def get_prediction_details(prediction_id: int):
    """Get detailed prediction information"""
    return predict_service.get_prediction_details(prediction_id)