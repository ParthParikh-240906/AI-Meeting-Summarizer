"""
Audio Transcription Module
Uses Groq Whisper Large v3 Turbo with audio chunking for large files
"""
import os
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB per chunk


class MeetingTranscriber:
    """
    Groq Whisper transcription with automatic chunking.
    Files > 25MB are split into chunks, transcribed separately, then combined.
    No local Whisper needed.
    """
    
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = "whisper-large-v3-turbo"
        print(f"Transcriber initialized: Groq {self.model_name} with chunking")
    
    def load_model(self):
        pass  # API is always ready
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio - chunks if > 25MB"""
        file_size = os.path.getsize(audio_path)
        
        if file_size <= GROQ_MAX_FILE_SIZE:
            return self._transcribe_single(audio_path)
        else:
            return self._transcribe_chunked(audio_path, file_size)
    
    def _transcribe_single(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe a single file via Groq"""
        print(f"Transcribing with Groq Whisper...")
        start_time = time.time()
        
        with open(audio_path, "rb") as f:
            response = self.client.audio.transcriptions.create(
                model=self.model_name,
                file=f,
                response_format="verbose_json",
            )
        
        return self._format_response(response, time.time() - start_time, "groq_single")
    
    def _transcribe_chunked(self, audio_path: str, file_size: int) -> Dict[str, Any]:
        """Split audio into <25MB chunks, transcribe each, combine"""
        num_chunks = (file_size // GROQ_MAX_FILE_SIZE) + 1
        print(f"File too large ({file_size/1024/1024:.1f}MB). Splitting into {num_chunks} chunks...")
        start_time = time.time()
        
        # Get audio duration
        duration = self._get_duration(audio_path)
        chunk_duration = duration / num_chunks
        
        chunks_dir = tempfile.mkdtemp()
        transcripts = []
        total_duration = 0
        
        try:
            for i in range(num_chunks):
                start_sec = i * chunk_duration
                chunk_path = os.path.join(chunks_dir, f"chunk_{i}.mp3")
                
                # Split with FFmpeg
                subprocess.run([
                    "ffmpeg", "-y", "-i", audio_path,
                    "-ss", str(start_sec), "-t", str(chunk_duration),
                    "-acodec", "libmp3lame", "-ab", "64k",
                    chunk_path
                ], capture_output=True)
                
                # Transcribe chunk
                with open(chunk_path, "rb") as f:
                    response = self.client.audio.transcriptions.create(
                        model=self.model_name,
                        file=f,
                        response_format="verbose_json",
                    )
                
                transcripts.append(response.text)
                total_duration += getattr(response, 'duration', chunk_duration)
                
                print(f"  Chunk {i+1}/{num_chunks} done")
        
        finally:
            # Cleanup temp files
            import shutil
            shutil.rmtree(chunks_dir, ignore_errors=True)
        
        full_text = " ".join(transcripts)
        
        return {
            "text": full_text,
            "segments": [],
            "language": "en",
            "duration": total_duration,
            "processing_time": time.time() - start_time,
            "method": f"groq_chunked_{num_chunks}"
        }
    
    def _get_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds using FFmpeg"""
        result = subprocess.run([
            "ffmpeg", "-i", audio_path
        ], capture_output=True, text=True)
        
        for line in result.stderr.split('\n'):
            if "Duration" in line:
                time_str = line.split("Duration: ")[1].split(",")[0]
                h, m, s = time_str.split(":")
                return float(h) * 3600 + float(m) * 60 + float(s)
        return 60  # Default fallback
    
    def _format_response(self, response, processing_time: float, method: str) -> Dict[str, Any]:
        """Format Groq response to standard dict"""
        segments = []
        if hasattr(response, 'segments'):
            for seg in response.segments:
                segments.append({
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": seg.get("text", "").strip()
                })
        
        return {
            "text": response.text,
            "segments": segments,
            "language": getattr(response, 'language', 'en'),
            "duration": getattr(response, 'duration', segments[-1]["end"] if segments else 0),
            "processing_time": processing_time,
            "method": method
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "type": "Groq Cloud with Audio Chunking",
            "max_chunk_size": "25MB",
            "no_local_fallback": True
        }