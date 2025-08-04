from typing import List, Dict, Any
from fastapi import HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from database.client import database_client
from services.predictions import predict_service
from datetime import datetime

class ChatService:
    def __init__(self):
        self.llm = self._init_gemini()
        self.database_client = database_client
    
    def _init_gemini(self) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7
        )
    
    def _get_system_prompt(self, prediction_data: Dict[str, Any]) -> str:
        prediction = prediction_data.get("prediction", {})
        class_data = prediction_data.get("class", {})
        
        predicted_class = class_data.get("name", "unknown")
        confidence = prediction.get("confidence", 0)
        description = class_data.get("description", "")
        
        return f"""Sen arı sağlığı konusunda uzman bir asistansın ve arıcıların kovan analiz sonuçlarını anlamalarına yardımcı oluyorsun.
Tahmin Detayları:

Tahmin edilen durum: {predicted_class}
Güven düzeyi: {confidence:.2%}
Açıklama: {description}

Rolün şunlardır:

Arı sağlığı durumunu basit, uygulanabilir terimlerle açıklamak
Arıcılar için pratik tavsiyeler sunmak
Gerekirse sonraki adımları veya tedavileri önermek
Arı sağlığı ve kovan yönetimi hakkındaki soruları yanıtlamak
Doğru bilgi verirken cesaretlendirici ve destekleyici olmak

Yanıtları faydalı, öz ve pratik arıcılık tavsiyeleri üzerine odaklanmış tut. Tavsiye verirken her zaman güven düzeyini dikkate al - güven düzeyi düşükse, belirsizlikten bahset ve daha fazla inceleme öner.
"""

    def create_chat(self, prediction_id: int) -> Dict[str, Any]:
        prediction_data = predict_service.get_prediction_details(prediction_id)
        if not prediction_data:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        chat_id = self.database_client.create_chat(prediction_id)
        
        system_prompt = self._get_system_prompt(prediction_data)
        self.database_client.create_message(chat_id, "system", system_prompt)
        
        predicted_class = prediction_data.get("class", {}).get("name", "unknown")
        class_description = prediction_data.get("class", {}).get("description", "")
        confidence = prediction_data.get("prediction", {}).get("confidence", 0)
        
        welcome_message = f"""Merhaba! Kovanız görüntüsünü analiz ettim ve şunu tespit ettim: {predicted_class} (güven: {confidence:.1%}).
{class_description}
Bu sonucu anlamanızda size yardımcı olmak ve kovan yönetiminiz için rehberlik sağlamak için buradayım. Şu konularda bana istediğiniz soruyu sorabilirsiniz:
Önerilen tedaviler veya eylemler
Önleme stratejileri
Genel arıcılık tavsiyeleri

Ne öğrenmek istiyorsunuz?"""
        
        self.database_client.create_message(chat_id, "assistant", welcome_message)
        
        return {
            "chat_id": chat_id,
            "prediction_id": prediction_id,
            "status": "created"
        }
    
    def add_message(self, chat_id: int, content: str) -> Dict[str, Any]:
        chat = self.database_client.get_chat(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        user_message_id = self.database_client.create_message(chat_id, "user", content)
        
        messages = self.database_client.get_messages_by_chat(chat_id)
        
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        
        try:
            response = self.llm(langchain_messages)
            ai_response = response.content
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")
        
        ai_message_id = self.database_client.create_message(chat_id, "assistant", ai_response)
        
        return {
            "id": ai_message_id,
            "chat_id": chat_id,
            "content": ai_response,
            "role": "assistant",
            "created_at": datetime.now().isoformat()
        }
    
    def get_chat_messages(self, chat_id: int) -> List[Dict[str, Any]]:
        chat = self.database_client.get_chat(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        messages = self.database_client.get_messages_by_chat(chat_id)
        
        user_messages = [
            msg for msg in messages 
            if msg["role"] in ["user", "assistant"]
        ]
        
        return user_messages
    
    def get_chat_details(self, chat_id: int) -> Dict[str, Any]:
        chat = self.database_client.get_chat(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        prediction_data = predict_service.get_prediction_details(chat["prediction_id"])
        messages = self.get_chat_messages(chat_id)
        
        return {
            "chat": chat,
            "prediction": prediction_data,
            "messages": messages,
            "message_count": len(messages)
        }


chat_service = ChatService()
