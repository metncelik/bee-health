from fastapi import UploadFile, File

class PredictService:
    def __init__(self):
        pass

    def predict(self, image: UploadFile = File(...)):
        return "a"
    
    
predict_service = PredictService()