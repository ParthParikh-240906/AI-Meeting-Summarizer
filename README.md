# Meeting Summary & Action Tracker

AI-powered meeting transcription, summarization, and action item extraction.
Built for Elchai Group AI Agent Intern Assessment.
100% Free | All Groq Cloud | Human-in-the-Loop Review

Website: https://meeting-tracker-bsl1.onrender.com

## Quick Start

1. Clone the repo
2. pip install -r requirements.txt
3. Add your Groq API key to .env: GROQ_API_KEY=gsk_your_key_here
4. Install FFmpeg (brew install ffmpeg on macOS)
5. python -m app.main
6. Open http://localhost:8000

## What It Does

- Upload meeting audio (any format, up to 300MB)
- Auto-converts to optimized MP3 format
- AI transcribes, summarizes, and extracts action items
- Large files automatically split into chunks for processing
- Human reviews and approves/rejects each action item
- Full audit trail logged automatically

## Tech Stack

- Backend: FastAPI (Python)
- Transcription: Groq Whisper Large v3 Turbo (with audio chunking)
- Summarization: Groq Llama 3.1 8B Instant
- Action Extraction: Groq Llama 3.1 8B Instant
- Frontend: HTML + CSS + Vanilla JavaScript
- Audio Processing: FFmpeg (format optimization + chunking)
- Cost: $0 (Groq free tier)

## Performance

- Small files: ~4 seconds total
- Large files: ~15-50 seconds (auto-chunked, fully on Groq)
- Max file size: 300MB upload, 74MB after optimization

## How It Works

1. Upload any audio format → FFmpeg converts to optimized MP3 (16kHz, 32kbps mono)
2. File < 25MB → Direct to Groq Whisper
3. File > 25MB → Split into chunks → Each chunk to Groq Whisper → Transcripts combined
4. Transcript → Groq Llama 3.1 for summarization + action extraction
5. All output marked "Pending Human Review" until approved/rejected

## Project Structure

app/           - FastAPI backend
  models/      - AI models (transcriber, summarizer, action extractor)
  services/    - Pipeline orchestrator, audit logger
  schemas/     - Pydantic data models
  utils/       - File handler + audio optimization
static/        - Web frontend
logs/          - Audit trail and review queue
test_data/     - Temporary uploaded files

## API Endpoints

GET  /                    - Web interface
GET  /docs                - Swagger API docs
GET  /health              - System health check
POST /process-meeting/    - Upload and process audio
GET  /audit-trail/        - View processing logs
GET  /statistics/         - System statistics

## Requirements

- Python 3.8+
- FFmpeg
- Groq API key (free from console.groq.com)

## Notes

- Files < 1MB rejected (too short for meaningful analysis)
- Max upload 300MB (compresses to 74MB max after optimization)
- Audio auto-converted to MP3 16kHz 32kbps mono for optimal processing
- 3-second cooldown between uploads prevents context leakage
- All AI output requires human review before finalizing
- No local models needed - everything runs on Groq Cloud
