# Meeting Summary & Action Tracker

AI-powered meeting transcription, summarization, and action item extraction.
Built for Elchai Group AI Agent Intern Assessment.
100% Free | Hybrid Architecture | Human-in-the-Loop Review

Website : https://meeting-tracker-bsl1.onrender.com 

## Quick Start

1. Clone the repo
2. pip install -r requirements.txt
3. Add your Groq API key to .env: GROQ_API_KEY=gsk_your_key_here
4. Install FFmpeg (brew install ffmpeg on macOS)
5. python -m app.main
6. Open http://localhost:8000

## What It Does

- Upload meeting audio (MP3/WAV/M4A)
- AI transcribes, summarizes, and extracts action items
- Human reviews and approves/rejects each action item
- Full audit trail logged automatically

## Tech Stack

- Backend: FastAPI (Python)
- Transcription: Groq Whisper Large v3 Turbo / Local Whisper base (fallback)
- Summarization: Groq Llama 3.1 8B Instant
- Frontend: HTML + CSS + Vanilla JavaScript
- Cost: $0 (Groq free tier)

## Performance

Small files (<25MB): ~4 seconds total (Groq Cloud)
Large files (>25MB): ~86 seconds total (Local Whisper + Groq)

## Project Structure

app/           - FastAPI backend
  models/      - AI models (transcriber, summarizer, action extractor)
  services/    - Pipeline orchestrator, audit logger
  schemas/     - Pydantic data models
  utils/       - File handler
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

- Files <1MB rejected (too short)
- Files >25MB use local Whisper (slower but handles any size)
- 3-second cooldown between uploads prevents context leakage
- All AI output requires human review before finalizing

Built for Elchai Group Assessment - August 2026
