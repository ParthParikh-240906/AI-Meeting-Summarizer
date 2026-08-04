"""
Pydantic schemas for Meeting Summary & Action Tracker
Defines data models for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class PriorityLevel(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class ReviewStatus(str, Enum):
    PENDING = "Pending Human Review"
    APPROVED = "Approved"
    REJECTED = "Rejected"
    MODIFIED = "Modified by Human"


class ActionItem(BaseModel):
    """Individual action item extracted from meeting"""
    task: str = Field(..., description="The action item description")
    assignee: Optional[str] = Field(None, description="Person responsible")
    deadline: Optional[str] = Field(None, description="Due date if mentioned")
    priority: PriorityLevel = Field(PriorityLevel.MEDIUM, description="Auto-assigned priority")
    status: ReviewStatus = Field(ReviewStatus.PENDING, description="Review status")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="AI confidence score")
    source_text: Optional[str] = Field(None, description="Original text that generated this item")


class MeetingMetadata(BaseModel):
    """Meeting metadata"""
    filename: str
    duration_seconds: float
    transcription_model: str
    summarization_model: str
    processed_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class MeetingSummary(BaseModel):
    """Complete meeting summary with action items"""
    id: Optional[str] = None
    title: str = Field("Untitled Meeting", description="Meeting title")
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    duration: str = "Unknown"
    participants: List[str] = Field(default_factory=list)
    
    # Summary content
    summary: str = Field("", description="Overall meeting summary")
    key_points: List[str] = Field(default_factory=list, description="Key discussion points")
    decisions: List[str] = Field(default_factory=list, description="Decisions made")
    
    # Action items
    action_items: List[ActionItem] = Field(default_factory=list)
    
    # Status tracking
    pending_review: bool = Field(True, description="Whether human review is needed")
    reviewer_status: ReviewStatus = Field(ReviewStatus.PENDING)
    reviewer_notes: Optional[str] = None
    
    # Metadata
    metadata: Optional[MeetingMetadata] = None
    full_transcript: Optional[str] = None


class AuditLogEntry(BaseModel):
    """Single audit log entry"""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    model_used: str
    tool_used: str
    prompt: str
    input_type: str
    output_type: str
    reviewer_status: str
    confidence_score: Optional[float] = None
    processing_time_seconds: Optional[float] = None
    error: Optional[str] = None


class ProcessingResponse(BaseModel):
    """API response after processing a meeting"""
    success: bool
    message: str
    summary: Optional[MeetingSummary] = None
    review_required: bool = True
    errors: List[str] = Field(default_factory=list)


class ReviewUpdate(BaseModel):
    """Human review update for action items"""
    action_item_id: int
    task: Optional[str] = None
    assignee: Optional[str] = None
    deadline: Optional[str] = None
    priority: Optional[PriorityLevel] = None
    status: ReviewStatus
    reviewer_notes: Optional[str] = None


class HealthCheck(BaseModel):
    """Health check response"""
    status: str = "healthy"
    version: str
    models_available: Dict[str, bool]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())