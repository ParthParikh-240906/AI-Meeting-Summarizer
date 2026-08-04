"""
Audio Transcription Module
Hybrid: Groq Cloud (fast) + Local Whisper (for large files)
"""
import os
import time
import whisper
from typing import Dict, Any
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# File size limit for Groq (25MB)
GROQ_MAX_FILE_SIZE = 25 * 1024 * 1024


class MeetingTranscriber:
    """
    Smart transcriber that switches between Groq and local Whisper.
    - Files < 25MB: Groq Whisper Large v3 Turbo (1-3 seconds)
    - Files > 25MB: Local Whisper base (2-5 minutes, but handles any size)
    """
    
    def __init__(self):
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.groq_model = "whisper-large-v3-turbo"
        self.model_name = self.groq_model
        self.local_model_size = "base"
        self.local_model = None
        print(f"Transcriber initialized (Hybrid: Groq + Local Whisper fallback)")
    
    def load_model(self):
        """Lazy load local Whisper only when needed"""
        if self.local_model is None:
            print("Loading local Whisper model (fallback for large files)...")
            self.local_model = whisper.load_model(self.local_model_size)
            print("Local Whisper ready")
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio - automatically chooses best method.
        """
        # Check file size
        file_size = os.path.getsize(audio_path)
        
        if file_size < GROQ_MAX_FILE_SIZE:
            return self._transcribe_groq(audio_path)
        else:
            print(f"⚠️ File too large for Groq ({file_size/1024/1024:.1f}MB), using local Whisper...")
            return self._transcribe_local(audio_path)
    
    def _transcribe_groq(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe using Groq cloud API (fast, free)"""
        print(f"Transcribing with Groq Whisper Large v3 Turbo...")
        start_time = time.time()
        
        with open(audio_path, "rb") as audio_file:
            response = self.groq_client.audio.transcriptions.create(
                model=self.groq_model,
                file=audio_file,
                response_format="verbose_json",
            )
        
        processing_time = time.time() - start_time
        
        segments = []
        if hasattr(response, 'segments'):
            for seg in response.segments:
                segments.append({
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": seg.get("text", "").strip()
                })
        
        output = {
            "text": response.text,
            "segments": segments,
            "language": getattr(response, 'language', 'en'),
            "duration": getattr(response, 'duration', segments[-1]["end"] if segments else 0),
            "processing_time": processing_time,
            "method": "groq_cloud"
        }
        
        print(f"✅ Groq transcription complete in {processing_time:.2f}s")
        return output
    
    def _transcribe_local(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe using local Whisper (slower but no size limit)"""
        self.load_model()
        
        print(f"Transcribing with local Whisper ({self.local_model_size})...")
        print("This may take 2-5 minutes for long audio...")
        start_time = time.time()
        
        result = self.local_model.transcribe(
            audio_path,
            fp16=False,
            verbose=False
        )
        
        processing_time = time.time() - start_time
        
        segments = []
        for segment in result.get("segments", []):
            segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip()
            })
        
        output = {
            "text": result["text"],
            "segments": segments,
            "language": result.get("language", "en"),
            "duration": result.get("duration", segments[-1]["end"] if segments else 0),
            "processing_time": processing_time,
            "method": "local_whisper"
        }
        
        print(f"✅ Local transcription complete in {processing_time:.2f}s")
        return output
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "primary": f"Groq/{self.groq_model}",
            "fallback": f"Local Whisper {self.local_model_size}",
            "switch_threshold": "25MB",
            "type": "Hybrid (Cloud + Local)"
        }