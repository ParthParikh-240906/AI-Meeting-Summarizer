"""
Processing Pipeline Service
Orchestrates the complete workflow: Transcription → Summarization → Action Extraction
"""
import re
import time
import uuid
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from groq import Groq

from app.models.transcriber import MeetingTranscriber
from app.models.summarizer import MeetingSummarizer
from app.models.action_extractor import ActionExtractor
from app.services.logger import audit_logger
from app.schemas.meeting import (
    MeetingSummary,
    MeetingMetadata,
    ActionItem,
    ProcessingResponse
)


class MeetingPipeline:
    """
    Complete meeting processing pipeline.
    Orchestrates all AI steps and manages the workflow.
    """
    
    def __init__(self):
        self.transcriber = MeetingTranscriber()
        self.summarizer = MeetingSummarizer()
        self.action_extractor = ActionExtractor()
        self.logger = audit_logger
        
        print("Meeting Pipeline initialized")
        print(f"  - Transcriber: {self.transcriber.model_name}")
        print(f"  - Summarizer: {self.summarizer.model_name}")
        print(f"  - Action Extractor: {self.action_extractor.model_name}")
    
    async def process_meeting(self, audio_file_path: str) -> ProcessingResponse:
        """
        Process a meeting audio file through the complete pipeline.
        
        Steps:
        1. Transcribe audio → text
        2. Generate meeting summary
        3. Extract action items
        4. Log all steps to audit trail
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            ProcessingResponse with full meeting summary and action items
        """
        meeting_id = str(uuid.uuid4())
        total_start_time = time.time()
        errors = []
        
        print(f"\n{'='*60}")
        print(f"Processing Meeting: {meeting_id}")
        print(f"Audio File: {audio_file_path}")
        print(f"{'='*60}\n")
        
        try:
            # ===========================================
            # STEP 1: TRANSCRIPTION
            # ===========================================
            print("📝 STEP 1: Transcribing audio...")
            step1_start = time.time()
            
            transcription_result = self.transcriber.transcribe(audio_file_path)
            transcript_text = transcription_result["text"]
            
            step1_time = time.time() - step1_start
            
            # Log step 1
            self.logger.log_processing_step(
                model_used=self.transcriber.model_name,
                tool_used="Groq Whisper Large v3 Turbo",
                prompt="Transcribe meeting audio with timestamps",
                input_type="audio/mpeg",
                output_type="text/transcript",
                reviewer_status="Auto-completed",
                confidence_score=0.95,
                processing_time_seconds=step1_time
            )
            
            print(f"✅ Transcription complete in {step1_time:.2f}s")
            print(f"   Transcript length: {len(transcript_text)} chars\n")

                        # Check if audio is too short for meaningful analysis
            audio_duration = transcription_result.get("duration", 0)
            transcript_length = len(transcript_text)
            
            if audio_duration < 30 or transcript_length < 200:  # Less than 30 sec or 200 chars
                print("⚠️ Audio too short - skipping AI analysis")

                metadata = MeetingMetadata(
                    filename=Path(audio_file_path).name,
                    duration_seconds=audio_duration,
                    transcription_model="Groq Whisper Large v3 Turbo",
                    summarization_model="skipped",
                    processed_at=datetime.now().isoformat()
                )
                
                meeting_summary = MeetingSummary(
                    id=meeting_id,
                    title="Short Audio - No Analysis",
                    date=datetime.now().strftime("%Y-%m-%d"),
                    duration=f"{audio_duration / 60:.1f} minutes",
                    participants=["Speaker only"],
                    summary=transcript_text,  # Summary IS the transcript
                    key_points=[],  # No key points
                    decisions=[],  # No decisions
                    action_items=[],  # No action items
                    pending_review=False,
                    reviewer_status="Auto-skipped (Audio too short)",
                    metadata=metadata,
                    full_transcript=transcript_text
                )
                
                return ProcessingResponse(
                    success=True,
                    message="Audio too short for analysis. Transcript displayed as summary.",
                    summary=meeting_summary,
                    review_required=False,
                    errors=[]
                )
            
            # ===========================================
            # STEP 2: SUMMARIZATION
            # ===========================================
            print("📄 STEP 2: Generating summary...")
            step2_start = time.time()
            
            summary_result = self.summarizer.summarize(transcript_text)
            
            step2_time = time.time() - step2_start
            
            # Log step 2
            self.logger.log_processing_step(
                model_used=self.summarizer.model_name,
                tool_used="Groq GPT-OSS 20B",
                prompt="Generate meeting summary with key points and decisions",
                input_type="text/transcript",
                output_type="json/summary",
                reviewer_status="Pending Human Review",
                confidence_score=0.85,
                processing_time_seconds=step2_time
            )
            
            print(f"✅ Summary generated in {step2_time:.2f}s")
            print(f"   Title: {summary_result['title']}\n")
            
            # ===========================================
            # STEP 3: ACTION ITEM EXTRACTION
            # ===========================================
            print("✅ STEP 3: Extracting action items...")
            step3_start = time.time()
            
            action_items_raw = self.action_extractor.extract(
                transcript_text,
                summary_result.get("summary", "")
            )
            
            step3_time = time.time() - step3_start
            
            # Convert to ActionItem models
            action_items = []
            for item in action_items_raw:
                action_items.append(
                    ActionItem(
                        task=item["task"],
                        assignee=item.get("assignee"),
                        deadline=item.get("deadline"),
                        priority=item.get("priority", "Medium"),
                        status="Pending Human Review",
                        confidence=item.get("confidence", 0.5),
                        source_text=item.get("task", "")[:200]
                    )
                )
            
            # Log step 3
            self.logger.log_processing_step(
                model_used=self.action_extractor.model_name,
                tool_used="Groq GPT-OSS 20B",
                prompt="Extract action items with assignees, deadlines, and priorities",
                input_type="text/transcript",
                output_type="json/action_items",
                reviewer_status="Pending Human Review",
                confidence_score=0.85,
                processing_time_seconds=step3_time
            )
            
            print(f"✅ Extracted {len(action_items)} action items in {step3_time:.2f}s\n")
            
            # ===========================================
            # BUILD FINAL RESPONSE
            # ===========================================
            total_time = time.time() - total_start_time
            
            # Extract participants (simple name detection)
            participants = self._extract_participants(transcript_text)
            
            # Create meeting metadata
            metadata = MeetingMetadata(
                filename=Path(audio_file_path).name,
                duration_seconds=transcription_result.get("duration", 0),
                transcription_model="Groq Whisper Large v3 Turbo",
                summarization_model=self.summarizer.model_name,
                processed_at=datetime.now().isoformat()
            )
            
            # Build the complete summary
            meeting_summary = MeetingSummary(
                id=meeting_id,
                title=summary_result["title"],
                date=datetime.now().strftime("%Y-%m-%d"),
                duration=f"{transcription_result.get('duration', 0) / 60:.1f} minutes",
                participants=participants,
                summary=summary_result["summary"],
                key_points=summary_result.get("key_points", []),
                decisions=summary_result.get("decisions", []),
                action_items=action_items,
                pending_review=True,
                reviewer_status="Pending Human Review",
                metadata=metadata,
                full_transcript=transcript_text[:5000]  # Store first 5000 chars
            )
            
            # Add to review queue
            self.logger.add_to_review_queue(
                meeting_id,
                [item.model_dump() for item in action_items]
            )
            
            print(f"{'='*60}")
            print(f"✅ PIPELINE COMPLETE in {total_time:.2f}s")
            print(f"   Steps: 3 AI processing steps")
            print(f"   Items pending review: {len(action_items)}")
            print(f"{'='*60}\n")
            
            return ProcessingResponse(
                success=True,
                message=f"Meeting processed successfully. {len(action_items)} action items need review.",
                summary=meeting_summary,
                review_required=True,
                errors=[]
            )
            
        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            print(f"❌ {error_msg}")
            errors.append(error_msg)
            
            # Log the error
            self.logger.log_processing_step(
                model_used="Pipeline",
                tool_used="MeetingPipeline",
                prompt="Process meeting end-to-end",
                input_type="audio/mpeg",
                output_type="json/meeting_summary",
                reviewer_status="Error",
                error=error_msg
            )
            
            return ProcessingResponse(
                success=False,
                message="Failed to process meeting",
                summary=None,
                review_required=False,
                errors=errors
            )
    
    def _extract_participants(self, transcript: str) -> list:
        """Extract participant names using Groq"""
        try:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))

            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{
                    "role": "user",
                    "content": f"""List ONLY the names of individual PEOPLE speaking in this conversation.
Exclude company names, brand names, and business names.
Return ONLY a JSON array of strings, like: ["FirstName LastName"]
No markdown, no explanation.

If you can't identify individual people, return ["Speaker 1"].

Transcript:
{transcript[:2000]}"""
                }],
                temperature=0.3,
                max_completion_tokens=512,
                reasoning_effort="low",
            )

            message = response.choices[0].message
            text = (getattr(message, "content", None) or "").strip()
            if not text:
                text = (getattr(message, "reasoning", None) or "").strip()

            participants = []
            try:
                participants = json.loads(text)
            except Exception:
                match = re.search(r"\[.*\]", text, re.DOTALL)
                if match:
                    try:
                        participants = json.loads(match.group())
                    except Exception:
                        participants = []

            if not isinstance(participants, list):
                participants = []

            # Filter obvious company/brand names
            company_words = [
                "inc", "corp", "llc", "ltd", "company", "store", "shop", "boutique"
            ]
            filtered = [
                p for p in participants
                if isinstance(p, str) and not any(cw in p.lower() for cw in company_words)
            ]

            if not filtered:
                return ["Speaker 1"]
            return filtered[:5]

        except Exception as e:
            print(f"Participant extraction error: {str(e)}")
            return ["Speaker 1"]


    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration"""
        return {
            "pipeline_version": "1.0.0",
            "steps": [
                {
                    "step": 1,
                    "name": "Transcription",
                    "model": self.transcriber.get_model_info(),
                    "status": "Available"
                },
                {
                    "step": 2,
                    "name": "Summarization",
                    "model": self.summarizer.get_model_info(),
                    "status": "Available"
                },
                {
                    "step": 3,
                    "name": "Action Extraction",
                    "model": self.action_extractor.get_model_info(),
                    "status": "Available"
                }
            ],
            "total_models": 3,
            "all_local": True,
            "requires_api_keys": False,
            "cost_per_meeting": "$0.00 (Free & Open Source)"
        }


# Singleton instance
pipeline = MeetingPipeline()


if __name__ == "__main__":
    import asyncio
    
    async def test_pipeline():
        # Test with sample file
        test_file = "test_data/sample_meeting.mp3"
        if Path(test_file).exists():
            result = await pipeline.process_meeting(test_file)
            print("\n" + "="*60)
            print("FINAL RESULT:")
            print(f"Success: {result.success}")
            print(f"Message: {result.message}")
            if result.summary:
                print(f"Title: {result.summary.title}")
                print(f"Summary: {result.summary.summary[:200]}...")
                print(f"Action Items: {len(result.summary.action_items)}")
        else:
            print(f"Add a test audio file at: {test_file}")
    
    asyncio.run(test_pipeline())