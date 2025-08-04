import io
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from services.speech import speech_service

router = APIRouter(
    prefix="/speech",
    tags=["speech"],
)


class TextToSpeechRequest(BaseModel):
    text: str
    volume: Optional[float] = 1.0
    length_scale: Optional[float] = 1.0  # Controls speech speed (higher = slower)
    noise_scale: Optional[float] = 0.667  # Controls audio variation
    noise_w_scale: Optional[float] = 0.8  # Controls speaking variation
    normalize_audio: Optional[bool] = True

class SpeechToTextResponse(BaseModel):
    text: str


@router.post("/text-to-speech")
async def text_to_speech(request: TextToSpeechRequest):
    try:
        audio_data = await speech_service.text_to_speech(
            text=request.text,
            volume=request.volume,
            length_scale=request.length_scale,
            noise_scale=request.noise_scale,
            noise_w_scale=request.noise_w_scale,
            normalize_audio=request.normalize_audio
        )
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/text-to-speech/stream")
async def text_to_speech_stream(request: TextToSpeechRequest):
    try:
        from piper import SynthesisConfig
        
        syn_config = SynthesisConfig(
            volume=request.volume,
            length_scale=request.length_scale,
            noise_scale=request.noise_scale,
            noise_w_scale=request.noise_w_scale,
            normalize_audio=request.normalize_audio
        )
        
        def generate_audio():
            for chunk in speech_service.synthesize_streaming(request.text, syn_config):
                yield chunk
                
        return StreamingResponse(
            generate_audio(),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech_stream.wav"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/speech-to-text", response_model=SpeechToTextResponse)
async def speech_to_text(audio_file: UploadFile = File(...)):
    try:
        if not audio_file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        text = speech_service.speech_to_text(audio_file.file)
        return SpeechToTextResponse(text=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))