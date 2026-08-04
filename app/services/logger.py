"""
Audit Logging Service
Tracks all AI processing steps with timestamps, model info, and review status
"""
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from app.config import AUDIT_LOG_FILE, REVIEW_QUEUE_FILE


class AuditLogger:
    """
    Comprehensive audit logging for AI operations.
    Tracks every AI step for transparency and review.
    """
    
    def __init__(self):
        self.log_file = AUDIT_LOG_FILE
        self.review_file = REVIEW_QUEUE_FILE
        self._ensure_log_files()
    
    def _ensure_log_files(self):
        """Create log files if they don't exist"""
        for file_path in [self.log_file, self.review_file]:
            if not file_path.exists():
                with open(file_path, 'w') as f:
                    json.dump([], f)
    
    def log_processing_step(
        self,
        model_used: str,
        tool_used: str,
        prompt: str,
        input_type: str,
        output_type: str,
        reviewer_status: str = "Pending Human Review",
        confidence_score: Optional[float] = None,
        processing_time_seconds: Optional[float] = None,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Log an AI processing step to the audit trail.
        
        Args:
            model_used: Name of the AI model (e.g., 'whisper-base')
            tool_used: Tool/library name (e.g., 'OpenAI Whisper')
            prompt: The prompt or input configuration used
            input_type: Type of input (e.g., 'audio/mp3', 'text/transcript')
            output_type: Type of output (e.g., 'text/transcript', 'json/summary')
            reviewer_status: Current review status
            confidence_score: AI confidence score (0.0 to 1.0)
            processing_time_seconds: Time taken to process
            error: Error message if any
        
        Returns:
            The log entry that was created
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_used": model_used,
            "tool_used": tool_used,
            "prompt": prompt[:500],  # Truncate long prompts
            "input_type": input_type,
            "output_type": output_type,
            "reviewer_status": reviewer_status,
            "confidence_score": confidence_score,
            "processing_time_seconds": processing_time_seconds,
            "error": error,
            "log_id": datetime.now().strftime("%Y%m%d%H%M%S%f")
        }
        
        # Read existing logs
        logs = self._read_logs()
        logs.append(log_entry)
        
        # Write back
        with open(self.log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        return log_entry
    
    def add_to_review_queue(self, meeting_id: str, action_items: List[Dict]) -> Dict[str, Any]:
        """
        Add meeting action items to the human review queue.
        
        Args:
            meeting_id: Unique meeting identifier
            action_items: List of action items needing review
        
        Returns:
            Review queue entry
        """
        review_entry = {
            "meeting_id": meeting_id,
            "submitted_at": datetime.now().isoformat(),
            "status": "Pending Human Review",
            "total_items": len(action_items),
            "items": []
        }
        
        for i, item in enumerate(action_items):
            review_entry["items"].append({
                "item_id": i,
                "task": item.get("task", ""),
                "assignee": item.get("assignee"),
                "deadline": item.get("deadline"),
                "priority": item.get("priority", "Medium"),
                "confidence": item.get("confidence", 0.0),
                "review_status": "Pending",
                "reviewer_notes": None,
                "modified_by_human": False
            })
        
        # Read existing queue
        queue = self._read_review_queue()
        queue.append(review_entry)
        
        # Write back
        with open(self.review_file, 'w') as f:
            json.dump(queue, f, indent=2)
        
        return review_entry
    
    def update_review_status(
        self,
        meeting_id: str,
        item_id: int,
        status: str,
        reviewer_notes: Optional[str] = None,
        modifications: Optional[Dict] = None
    ) -> bool:
        """
        Update the review status of an action item.
        
        Args:
            meeting_id: Meeting identifier
            item_id: Action item index
            status: New review status
            reviewer_notes: Notes from human reviewer
            modifications: Any modifications made by reviewer
        
        Returns:
            True if updated successfully
        """
        queue = self._read_review_queue()
        
        for entry in queue:
            if entry["meeting_id"] == meeting_id:
                for item in entry["items"]:
                    if item["item_id"] == item_id:
                        item["review_status"] = status
                        item["reviewer_notes"] = reviewer_notes
                        if modifications:
                            item.update(modifications)
                            item["modified_by_human"] = True
                        
                        # Update entry status
                        all_reviewed = all(
                            i["review_status"] != "Pending" 
                            for i in entry["items"]
                        )
                        entry["status"] = "Fully Reviewed" if all_reviewed else "Partially Reviewed"
                        
                        with open(self.review_file, 'w') as f:
                            json.dump(queue, f, indent=2)
                        
                        return True
        
        return False
    
    def get_audit_trail(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get the most recent audit log entries"""
        logs = self._read_logs()
        return logs[-limit:]
    
    def get_review_queue(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get review queue items, optionally filtered by status"""
        queue = self._read_review_queue()
        if status:
            return [entry for entry in queue if entry["status"] == status]
        return queue
    
    def get_logs_by_model(self, model_name: str) -> List[Dict[str, Any]]:
        """Get audit logs filtered by model"""
        logs = self._read_logs()
        return [log for log in logs if model_name.lower() in log["model_used"].lower()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics"""
        logs = self._read_logs()
        queue = self._read_review_queue()
        
        models_used = set(log["model_used"] for log in logs)
        
        return {
            "total_processing_steps": len(logs),
            "models_used": list(models_used),
            "pending_reviews": len([e for e in queue if e["status"] == "Pending Human Review"]),
            "completed_reviews": len([e for e in queue if e["status"] == "Fully Reviewed"]),
            "last_activity": logs[-1]["timestamp"] if logs else None
        }
    
    def _read_logs(self) -> List[Dict[str, Any]]:
        """Read audit logs from file"""
        try:
            with open(self.log_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _read_review_queue(self) -> List[Dict[str, Any]]:
        """Read review queue from file"""
        try:
            with open(self.review_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def clear_logs(self):
        """Clear all logs (use with caution)"""
        for file_path in [self.log_file, self.review_file]:
            with open(file_path, 'w') as f:
                json.dump([], f)
        print("All logs cleared")


# Singleton instance
audit_logger = AuditLogger()


if __name__ == "__main__":
    # Test the logger
    logger = AuditLogger()
    
    # Log a sample processing step
    logger.log_processing_step(
        model_used="whisper-base",
        tool_used="OpenAI Whisper",
        prompt="Transcribe meeting audio",
        input_type="audio/mp3",
        output_type="text/transcript",
        confidence_score=0.95,
        processing_time_seconds=2.5
    )
    
    # Add to review queue
    logger.add_to_review_queue(
        "meeting_001",
        [
            {"task": "Prepare report", "assignee": "John", "deadline": "Friday"},
            {"task": "Schedule meeting", "assignee": None, "deadline": "Next week"}
        ]
    )
    
    # Get statistics
    stats = logger.get_statistics()
    print("\n--- Audit Statistics ---")
    print(json.dumps(stats, indent=2))
    
    # Show recent logs
    print("\n--- Recent Audit Trail ---")
    for log in logger.get_audit_trail(5):
        print(f"[{log['timestamp']}] {log['model_used']} - {log['reviewer_status']}")