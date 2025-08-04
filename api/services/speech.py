from typing import BinaryIO, Optional
from fastapi import HTTPException
from piper import PiperVoice
from piper import SynthesisConfig
import whisper
import wave
import io
import os

import whisper.transcribe

class SpeechService:
    def __init__(self, use_cuda: bool = False):
        piper_model_path = "checkpoints/piper-tts/piper-tr.onnx"
        if os.path.exists(piper_model_path):
            self.voice = PiperVoice.load(piper_model_path, use_cuda=use_cuda)
        else:
            self.voice = None
        
        self.whisper_model = whisper.load_model("turbo")
    
    async def text_to_speech(
        self, 
        text: str, 
        volume: float = 1.0,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w_scale: float = 0.8,
        normalize_audio: bool = True
    ) -> bytes:
        if not self.voice:
            raise HTTPException(status_code=500, detail="Voice model not loaded")
        
        try:
            # Create synthesis configuration
            syn_config = SynthesisConfig(
                volume=volume,
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w_scale=noise_w_scale,
                normalize_audio=normalize_audio
            )
            
            # Create in-memory WAV file
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                self.voice.synthesize_wav(text, wav_file, syn_config=syn_config)
            
            # Return the WAV data
            wav_buffer.seek(0)
            return wav_buffer.read()
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {str(e)}")
    
    def synthesize_streaming(self, text: str, syn_config: Optional[SynthesisConfig] = None):
        """
        Generator for streaming audio synthesis
        """
        if not self.voice:
            raise HTTPException(status_code=500, detail="Voice model not loaded")
        
        try:
            for audio_chunk in self.voice.synthesize(text, syn_config=syn_config):
                yield audio_chunk.audio_int16_bytes
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Streaming synthesis failed: {str(e)}")
    
    def speech_to_text(self, audio_file: BinaryIO) -> str:
        if self.whisper_model is None:
            raise HTTPException(status_code=500, detail="Whisper model not loaded")

        try:
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name

            result = self.whisper_model.transcribe(tmp_path)
            text = result.get("text", "").strip()

            os.remove(tmp_path)

            return text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Speech-to-text failed: {str(e)}")


# Create a singleton instance
speech_service = SpeechService()