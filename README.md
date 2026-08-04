meeting-summary-tracker/
├── app/
│   ├── main.py              # FastAPI server
│   ├── config.py            # Settings
│   ├── models/
│   │   ├── transcriber.py   # Whisper (local)
│   │   ├── summarizer.py    # Groq → Llama 3.1 (cloud, free)
│   │   └── action_extractor.py  # Groq → Llama 3.1 (cloud, free)
│   ├── services/
│   │   ├── pipeline.py      # Orchestrator
│   │   └── logger.py        # Audit trail
│   ├── schemas/
│   │   └── meeting.py       # Data models
│   └── utils/
│       └── file_handler.py  # File management
├── static/
│   └── index.html           # Web UI
├── logs/                    # Audit logs
├── test_data/               # Uploaded files
├── .env                     # API keys
└── requirements.txt         # Dependencies