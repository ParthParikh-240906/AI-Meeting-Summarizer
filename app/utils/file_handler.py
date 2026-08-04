"""
File Handler Utility
Manages audio file uploads, validation, and cleanup
"""
import os
import shutil
import uuid
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime
from app.config import UPLOAD_DIR, ALLOWED_AUDIO_TYPES, MAX_FILE_SIZE


class FileHandler:
    """
    Handles file operations for meeting audio files.
    Validates, saves, and manages cleanup of uploaded files.
    """
    
    def __init__(self, upload_dir: Path = UPLOAD_DIR):
        self.upload_dir = upload_dir
        self.upload_dir.mkdir(exist_ok=True)
    
    def validate_file(self, filename: str, content_type: str, file_size: int) -> Tuple[bool, Optional[str]]:
        """
        Validate uploaded file.
        
        Args:
            filename: Original filename
            content_type: MIME type of file
            file_size: File size in bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check minimum file size (1MB)
        min_size = 1 * 1024 * 1024  # 1MB
        if file_size < min_size:
            return False, f"File too small: {file_size / 1024:.1f}KB. Minimum: 1MB"

        # Check content type
        if content_type not in ALLOWED_AUDIO_TYPES:
            return False, f"Invalid file type: {content_type}. Allowed: {', '.join(ALLOWED_AUDIO_TYPES)}"
        
        # Check file extension
        allowed_extensions = {'.mp3', '.wav', '.m4a', '.ogg', '.webm', '.mp4', '.mpeg'}
        file_ext = Path(filename).suffix.lower()
        if file_ext not in allowed_extensions:
            return False, f"Invalid file extension: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
        
        # Check file size
        if file_size > MAX_FILE_SIZE:
            max_mb = MAX_FILE_SIZE / (1024 * 1024)
            return False, f"File too large: {file_size / (1024 * 1024):.1f}MB. Maximum: {max_mb:.0f}MB"
        
        # Check if file is empty
        if file_size == 0:
            return False, "File is empty"
        
        return True, None
    
    def save_upload(self, file_data: bytes, original_filename: str) -> Tuple[str, str]:
        """
        Save uploaded file with unique name.
        
        Args:
            file_data: Raw file bytes
            original_filename: Original filename
            
        Returns:
            Tuple of (saved_filepath, unique_filename)
        """
        # Generate unique filename
        file_ext = Path(original_filename).suffix.lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        unique_filename = f"meeting_{timestamp}_{unique_id}{file_ext}"
        
        # Save file
        file_path = self.upload_dir / unique_filename
        with open(file_path, "wb") as f:
            f.write(file_data)
        
        print(f"File saved: {file_path}")
        print(f"Size: {len(file_data) / 1024:.1f} KB")
        
        return str(file_path), unique_filename
    
    def get_file_info(self, file_path: str) -> dict:
        """
        Get information about a saved file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        path = Path(file_path)
        
        if not path.exists():
            return {"error": "File not found"}
        
        return {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            "extension": path.suffix,
            "created": datetime.fromtimestamp(path.stat().st_ctime).isoformat(),
            "path": str(path.absolute())
        }
    
    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """
        Remove files older than specified hours.
        
        Args:
            max_age_hours: Maximum age of files to keep
            
        Returns:
            Number of files removed
        """
        removed_count = 0
        current_time = datetime.now().timestamp()
        max_age_seconds = max_age_hours * 3600
        
        for file_path in self.upload_dir.iterdir():
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_path.unlink()
                    removed_count += 1
                    print(f"Removed old file: {file_path.name}")
        
        return removed_count
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a specific file.
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            True if deleted successfully
        """
        path = Path(file_path)
        if path.exists() and path.is_file():
            path.unlink()
            print(f"Deleted: {path.name}")
            return True
        return False
    
    def list_uploads(self) -> list:
        """
        List all uploaded files.
        
        Returns:
            List of file information dictionaries
        """
        files = []
        for file_path in self.upload_dir.iterdir():
            if file_path.is_file():
                files.append(self.get_file_info(str(file_path)))
        return files
    
    def get_storage_stats(self) -> dict:
        """
        Get storage statistics for upload directory.
        
        Returns:
            Dictionary with storage information
        """
        total_size = 0
        file_count = 0
        
        for file_path in self.upload_dir.iterdir():
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
        
        return {
            "upload_directory": str(self.upload_dir.absolute()),
            "total_files": file_count,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024)
        }


# Singleton instance
file_handler = FileHandler()


if __name__ == "__main__":
    # Test file handler
    handler = FileHandler()
    
    # Test validation
    print("--- File Validation Tests ---")
    valid, error = handler.validate_file("test.mp3", "audio/mpeg", 1024 * 1024)
    print(f"Valid MP3: {valid}, Error: {error}")
    
    valid, error = handler.validate_file("test.txt", "text/plain", 1024)
    print(f"Invalid TXT: {valid}, Error: {error}")
    
    valid, error = handler.validate_file("large.mp3", "audio/mpeg", 100 * 1024 * 1024)
    print(f"Too large: {valid}, Error: {error}")
    
    # Test saving a small file
    test_data = b"This is test audio data"
    path, name = handler.save_upload(test_data, "test_meeting.mp3")
    print(f"\nSaved: {name} at {path}")
    
    # Get storage stats
    stats = handler.get_storage_stats()
    print(f"\n--- Storage Stats ---")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Cleanup test file
    handler.delete_file(path)
    print("\nTest file cleaned up")