import os
import torch
import numpy as np
from PIL import Image
from fastapi import UploadFile, File, HTTPException
from torchvision import transforms, models
import torch.nn as nn
import io
from pathlib import Path
import time
from database.client import database_client

BEE_HEALTH_CLASSES = [
    "ant problems",
    "few varroa, hive beetles",
    "healthy",
    "hive being robbed",
    "missing queen",
    "varroa, small hive beetles"
]


class PredictService:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self.class_names = BEE_HEALTH_CLASSES
        self._init_transform()
        self._init_classes_in_db()

    def _init_transform(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _init_classes_in_db(self):
        """Initialize bee health classes in database"""
        class_descriptions = {
            "ant problems": "Kovan karınca istilası sorunları yaşıyor",
            "few varroa, hive beetles": "Az sayıda varroa akarı ve küçük kovan böceği var",
            "healthy": "Kovan sağlıklı görünüyor, görünür bir sorun yok",
            "hive being robbed": "Kovan başka arılar veya böcekler tarafından yağmalanıyor",
            "missing queen": "Kovan kraliçe arısını kaybetmiş gibi görünüyor",
            "varroa, small hive beetles": "Belirgin miktarda varroa akarı ve küçük kovan böceği var"
        }

        for class_name in self.class_names:
            existing = database_client.get_class_by_name(class_name)
            if not existing:
                database_client.create_class(
                    class_name, class_descriptions[class_name])

    def load_model(self, model_name="vgg16-bee-health"):
        try:
            model_path = f"checkpoints/image-classifier/{model_name}.pth"
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False)

            num_classes = checkpoint.get('num_classes', 6)
            pretrained = checkpoint.get('pretrained', True)

            self.model = models.vgg16(pretrained=pretrained)
            self.model.classifier[6] = nn.Linear(
                self.model.classifier[6].in_features, num_classes)

            self.model.load_state_dict(checkpoint['model_state_dict'])

            if 'class_names' in checkpoint:
                self.class_names = checkpoint['class_names']

            self.model = self.model.to(self.device)
            self.model.eval()
            return True
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Error loading model: {str(e)}")

    def _process_image(self, image_data: bytes):
        try:
            image = Image.open(io.BytesIO(image_data))

            if image.mode != 'RGB':
                image = image.convert('RGB')

            image_tensor = self.transform(image)
            image_tensor = image_tensor.unsqueeze(0)

            return image_tensor.to(self.device)

        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Error processing image: {str(e)}")

    def _save_image(self, image_data: bytes, filename: str) -> str:
        images_dir = Path("storage/images/")
        images_dir.mkdir(parents=True, exist_ok=True)

        timestamp = str(int(time.time()))
        file_extension = filename.split('.')[-1] if '.' in filename else 'jpg'
        unique_filename = f"{timestamp}_{filename}.{file_extension}"

        file_path = images_dir / unique_filename

        with open(file_path, 'wb') as f:
            f.write(image_data)

        return str(file_path)

    async def predict(self, image: UploadFile = File(...)):
        try:
            if not image.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400, detail="File must be an image")

            image_data = await image.read()

            image_url = self._save_image(image_data, image.filename)

            image_id = database_client.create_image(image_url)

            if self.model is None:
                self.load_model()

            image_tensor = self._process_image(image_data)

            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_class_idx = torch.max(probabilities, 1)

                confidence_score = confidence.item()
                predicted_class = self.class_names[predicted_class_idx.item()]

            prediction_id = database_client.create_prediction(
                image_id, confidence_score)

            class_record = database_client.get_class_by_name(predicted_class)
            if class_record:
                database_client.link_prediction_to_class(
                    prediction_id, class_record['id'])

            return {
                "prediction_id": prediction_id,
                "image_id": image_id,
                "predicted_class": predicted_class,
                "confidence": confidence_score,
                "all_probabilities": {
                    self.class_names[i]: float(probabilities[0][i])
                    for i in range(len(self.class_names))
                }
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Prediction failed: {str(e)}")

    def get_prediction_details(self, prediction_id: int):
        prediction = database_client.get_prediction(prediction_id)
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")

        class_record = database_client.get_prediction_class(prediction_id)
        image_record = database_client.get_image(prediction['image_id'])
        chat_record = database_client.get_chat_by_prediction(prediction_id)

        return {
            "prediction": prediction,
            "class": class_record,
            "image": image_record,
            "chat": chat_record
        }

    def get_predictions(self):
        predictions = database_client.get_predictions()
        for prediction in predictions:
            class_record = database_client.get_prediction_class(
                prediction['id'])
            prediction['class'] = class_record

            image_record = database_client.get_image(prediction['image_id'])
            prediction['image'] = image_record

            chat_record = database_client.get_chat_by_prediction(
                prediction['id'])
            prediction['chat'] = chat_record

        return predictions


predict_service = PredictService()
