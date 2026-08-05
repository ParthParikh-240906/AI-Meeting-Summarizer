"""
FastAPI Main Application
Meeting Summary & Action Tracker API
100% Free - All models run locally
"""
import os
import time
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import APP_TITLE, APP_VERSION, DEBUG_MODE, UPLOAD_DIR
from app.services.pipeline import MeetingPipeline
from app.services.logger import audit_logger
from app.utils.file_handler import file_handler
from app.schemas.meeting import (
    ProcessingResponse,
    ReviewUpdate,
    HealthCheck,
    AuditLogEntry
)


# ============================================
# Initialize FastAPI App
# ============================================
app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description="AI-powered meeting summarization and action tracking - 100% Free & Local",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent.parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Initialize services
pipeline = MeetingPipeline()


# ============================================
# API Endpoints
# ============================================

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main web interface"""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return index_path.read_text()
    return """
    <html>
        <body>
            <h1>Meeting Summary & Action Tracker</h1>
            <p>API is running. Visit <a href="/docs">/docs</a> for API documentation.</p>
        </body>
    </html>
    """


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """
    Health check endpoint - verifies all systems are operational.
    """
    return HealthCheck(
        status="healthy",
        version=APP_VERSION,
        models_available={
            "whisper_transcription": True,
            "bart_summarization": True,
            "action_extraction": True
        },
        timestamp=datetime.now().isoformat()
    )


@app.post("/process-meeting/", response_model=ProcessingResponse)
async def process_meeting(file: UploadFile = File(...)):
    """
    Process a meeting audio file through the complete AI pipeline.
    
    Steps:
    1. Transcribe audio to text (Whisper)
    2. Generate meeting summary (BART)
    3. Extract action items (Hybrid NLP)
    4. Log everything to audit trail
    
    Returns meeting summary with action items pending human review.
    """
    start_time = time.time()
    
    try:
        # Validate file
        file_data = await file.read()
        is_valid, error_msg = file_handler.validate_file(
            file.filename,
            file.content_type or "audio/mpeg",
            len(file_data)
        )
        
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Save uploaded file
        file_path, saved_name = file_handler.save_upload(file_data, file.filename)
        
        # Process through pipeline
        result = await pipeline.process_meeting(file_path)
        
        # Clean up uploaded file after processing
        file_handler.delete_file(file_path)
        
        total_time = time.time() - start_time
        print(f"Total request time: {total_time:.2f}s")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing meeting: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    except Exception as e:
        error_str = str(e)
        if "Request Entity Too Large" in error_str or "413" in error_str:
            raise HTTPException(
                status_code=413, 
                detail="File too large for cloud demo (>25MB). Please run locally for large files."
            )


@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload audio file only (without processing).
    Useful for pre-loading files before processing.
    """
    try:
        file_data = await file.read()
        is_valid, error_msg = file_handler.validate_file(
            file.filename,
            file.content_type or "audio/mpeg",
            len(file_data)
        )
        
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        file_path, saved_name = file_handler.save_upload(file_data, file.filename)
        file_info = file_handler.get_file_info(file_path)
        
        return JSONResponse(content={
            "success": True,
            "filename": saved_name,
            "file_path": file_path,
            "file_info": file_info,
            "message": "File uploaded successfully. Use /process-meeting/ to process."
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/review-queue/")
async def get_review_queue(status: Optional[str] = None):
    """
    Get items pending human review.
    
    Args:
        status: Filter by review status (Pending, Reviewed, etc.)
    """
    queue = audit_logger.get_review_queue(status)
    return JSONResponse(content={
        "total": len(queue),
        "items": queue,
        "timestamp": datetime.now().isoformat()
    })


@app.post("/review/update/")
async def update_review(review: ReviewUpdate):
    """
    Update review status of an action item (Human-in-the-loop).
    
    Allows reviewers to:
    - Approve, reject, or modify action items
    - Add reviewer notes
    - Update task details
    """
    success = audit_logger.update_review_status(
        meeting_id=review.meeting_id if hasattr(review, 'meeting_id') else "unknown",
        item_id=review.action_item_id,
        status=review.status,
        reviewer_notes=review.reviewer_notes,
        modifications={
            "task": review.task,
            "assignee": review.assignee,
            "deadline": review.deadline,
            "priority": review.priority
        } if any([review.task, review.assignee, review.deadline, review.priority]) else None
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Review item not found")
    
    return JSONResponse(content={
        "success": True,
        "message": "Review updated successfully"
    })


@app.get("/audit-trail/")
async def get_audit_trail(limit: int = 50, model: Optional[str] = None):
    """
    Get the audit trail of all AI processing steps.
    
    Args:
        limit: Number of entries to return
        model: Filter by model name
    """
    if model:
        logs = audit_logger.get_logs_by_model(model)
    else:
        logs = audit_logger.get_audit_trail(limit)
    
    return JSONResponse(content={
        "total": len(logs),
        "logs": logs[-limit:],
        "timestamp": datetime.now().isoformat()
    })


@app.get("/statistics/")
async def get_statistics():
    """
    Get processing statistics and system information.
    """
    audit_stats = audit_logger.get_statistics()
    storage_stats = file_handler.get_storage_stats()
    pipeline_info = pipeline.get_pipeline_info()
    
    return JSONResponse(content={
        "system": {
            "app_name": APP_TITLE,
            "version": APP_VERSION,
            "models": pipeline_info["steps"],
            "cost_per_meeting": "$0.00 (Free & Open Source)"
        },
        "audit": audit_stats,
        "storage": storage_stats,
        "timestamp": datetime.now().isoformat()
    })


@app.get("/pipeline-info/")
async def get_pipeline_info():
    """
    Get detailed information about the AI pipeline.
    """
    return JSONResponse(content=pipeline.get_pipeline_info())


@app.delete("/cleanup/")
async def cleanup_files(max_age_hours: int = 24):
    """
    Clean up old uploaded files.
    """
    removed = file_handler.cleanup_old_files(max_age_hours)
    return JSONResponse(content={
        "success": True,
        "files_removed": removed,
        "message": f"Cleaned up {removed} files older than {max_age_hours} hours"
    })


# ============================================
# Error Handlers
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================
# Startup Event
# ============================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("\n" + "="*60)
    print(f"🚀 {APP_TITLE} v{APP_VERSION}")
    print("="*60)
    print(f"📁 Upload directory: {UPLOAD_DIR}")
    print(f"📁 Logs directory: {Path(__file__).parent.parent / 'logs'}")
    print(f"📁 Static files: {static_dir}")
    print(f"📊 API Docs: http://localhost:8000/docs")
    print(f"🖥️  Web UI: http://localhost:8000/")
    print(f"💵 Cost: $0.00 (100% Free & Local)")
    print("="*60 + "\n")


# ============================================
# Run Application
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("MEETING SUMMARY & ACTION TRACKER")
    print("100% Free | Local AI | No API Keys Needed")
    print("="*60)
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=DEBUG_MODE,
        log_level="info"
    )