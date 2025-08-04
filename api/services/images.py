import os
from fastapi import HTTPException
from fastapi.responses import FileResponse

class ImageService:
    def __init__(self):
        self.storage_path = "storage/images"
    
    def get_image(self, filename: str):
        """Get image file by filename"""
        file_path = os.path.join(self.storage_path, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Image not found")
        
        return FileResponse(
            path=file_path,
            media_type="image/png"
        )

image_service = ImageService()
