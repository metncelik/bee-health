from fastapi import APIRouter
from services.images import image_service

router = APIRouter(
    prefix="/storage/images",
    tags=["images"],
)

@router.get("/{filename}")
async def get_image(filename: str):
    """Get image by filename"""
    return image_service.get_image(filename)
