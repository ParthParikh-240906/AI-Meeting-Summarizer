"""
Configuration settings for Meeting Summary Tracker
Uses free, local models - no API keys needed
"""
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
UPLOAD_DIR = BASE_DIR / "test_data"
MODELS_CACHE = Path.home() / ".cache" / "meeting_tracker"

# Create directories if they don't exist
LOGS_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
MODELS_CACHE.mkdir(parents=True, exist_ok=True)

# Model configurations
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large
                        # 'base' is good balance of speed vs accuracy

SUMMARIZATION_MODEL = "facebook/bart-large-cnn"  # Free HuggingFace model

# Action extraction settings
ACTION_KEYWORDS = [
    "need to", "will", "should", "must", "going to",
    "action item", "task", "todo", "to-do",
    "responsible for", "assigned to", "deadline",
    "by tomorrow", "by next week", "by end of"
]

# Logging
AUDIT_LOG_FILE = LOGS_DIR / "audit_trail.json"
REVIEW_QUEUE_FILE = LOGS_DIR / "review_queue.json"

# File upload settings
ALLOWED_AUDIO_TYPES = [
    "audio/mpeg", "audio/mp3", "audio/wav", 
    "audio/x-wav",
    "audio/m4a","audio/x-m4a","audio/ogg", "audio/webm"
]

# Application settings
APP_TITLE = "Meeting Summary & Action Tracker"
APP_VERSION = "1.0.0"
DEBUG_MODE = True