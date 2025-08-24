"""
Helper utilities and common functions.
"""

import hashlib
import mimetypes
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiofiles

from ..models.schemas import ContentType


def generate_id(prefix: str = "") -> str:
    """Generate a unique identifier with optional prefix."""
    unique_id = str(uuid.uuid4())
    return f"{prefix}{unique_id}" if prefix else unique_id


def generate_hash(content: Union[str, bytes]) -> str:
    """Generate SHA-256 hash of content."""
    if isinstance(content, str):
        content = content.encode('utf-8')
    return hashlib.sha256(content).hexdigest()


def detect_content_type(file_path: str, mime_type: Optional[str] = None) -> ContentType:
    """Detect content type from file path and MIME type."""
    if mime_type is None:
        mime_type, _ = mimetypes.guess_type(file_path)
    
    if mime_type:
        if mime_type.startswith('image/'):
            return ContentType.IMAGE
        elif mime_type in ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            return ContentType.DOCUMENT
        elif mime_type.startswith('text/'):
            return ContentType.TEXT
    
    # Fallback to file extension
    ext = Path(file_path).suffix.lower()
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
        return ContentType.IMAGE
    elif ext in ['.pdf', '.docx']:
        return ContentType.DOCUMENT
    else:
        return ContentType.TEXT


def is_valid_file_extension(file_path: str, allowed_extensions: List[str]) -> bool:
    """Check if file has a valid extension."""
    ext = Path(file_path).suffix.lower().lstrip('.')
    return ext in [e.lower() for e in allowed_extensions]


def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(file_path)


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None
) -> List[str]:
    """
    Split text into chunks with overlap, respecting word boundaries.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        separators: List of separators to use for splitting (in order of preference)
    
    Returns:
        List of text chunks
    """
    if separators is None:
        separators = ['\n\n', '\n', '. ', ' ', '']
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Find the best place to split
        split_pos = end
        for separator in separators:
            if separator:
                pos = text.rfind(separator, start, end)
                if pos > start:
                    split_pos = pos + len(separator)
                    break
        
        chunk = text[start:split_pos].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = split_pos - chunk_overlap
        if start < 0:
            start = 0
    
    return [chunk for chunk in chunks if chunk.strip()]


def merge_dictionaries(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries, with later ones taking precedence."""
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing/replacing invalid characters."""
    import re
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'\s+', '_', filename)
    return filename[:255]  # Limit length


async def async_retry(
    func,
    *args,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple = (Exception,),
    **kwargs
):
    """
    Async retry decorator with exponential backoff.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch and retry
    """
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            if attempt == max_retries:
                raise
            await asyncio.sleep(delay * (backoff ** attempt))


def validate_embedding_dimension(embedding: List[float], expected_dim: int) -> bool:
    """Validate that embedding has the expected dimension."""
    return len(embedding) == expected_dim


def normalize_text(text: str) -> str:
    """Normalize text by removing extra whitespace and converting to lowercase."""
    import re
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text using simple frequency analysis."""
    import re
    from collections import Counter
    
    # Simple keyword extraction (can be enhanced with NLP libraries)
    words = re.findall(r'\b\w+\b', text.lower())
    # Filter out common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'
    }
    
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    word_counts = Counter(filtered_words)
    
    return [word for word, _ in word_counts.most_common(max_keywords)]


class AsyncJobTracker:
    """Track the status of async jobs."""
    
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
    
    def create_job(self, job_id: str, **kwargs) -> None:
        """Create a new job tracking entry."""
        self._jobs[job_id] = {
            'id': job_id,
            'status': 'pending',
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
            'progress': 0.0,
            **kwargs
        }
    
    def update_job(self, job_id: str, **kwargs) -> None:
        """Update job status and progress."""
        if job_id in self._jobs:
            self._jobs[job_id].update(kwargs)
            self._jobs[job_id]['updated_at'] = datetime.utcnow()
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status."""
        return self._jobs.get(job_id)
    
    def list_jobs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all jobs, optionally filtered by status."""
        jobs = list(self._jobs.values())
        if status:
            jobs = [job for job in jobs if job.get('status') == status]
        return jobs
    
    def cleanup_completed_jobs(self, max_age_hours: int = 24) -> int:
        """Remove completed jobs older than max_age_hours."""
        cutoff_time = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        to_remove = []
        
        for job_id, job in self._jobs.items():
            if (job.get('status') in ['completed', 'failed'] and 
                job.get('updated_at', datetime.utcnow()).timestamp() < cutoff_time):
                to_remove.append(job_id)
        
        for job_id in to_remove:
            del self._jobs[job_id]
        
        return len(to_remove)


class FileManager:
    """Utility for managing file operations."""
    
    @staticmethod
    async def save_upload_file(file_content: bytes, filename: str, upload_dir: str) -> str:
        """Save uploaded file and return the file path."""
        os.makedirs(upload_dir, exist_ok=True)
        safe_filename = sanitize_filename(filename)
        file_path = os.path.join(upload_dir, safe_filename)
        
        # Ensure unique filename
        counter = 1
        base_name, ext = os.path.splitext(safe_filename)
        while os.path.exists(file_path):
            file_path = os.path.join(upload_dir, f"{base_name}_{counter}{ext}")
            counter += 1
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)
        
        return file_path
    
    @staticmethod
    async def read_file_async(file_path: str) -> bytes:
        """Read file content asynchronously."""
        async with aiofiles.open(file_path, 'rb') as f:
            return await f.read()
    
    @staticmethod
    def cleanup_temp_files(temp_dir: str, max_age_hours: int = 24) -> int:
        """Clean up temporary files older than max_age_hours."""
        if not os.path.exists(temp_dir):
            return 0
        
        cutoff_time = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        removed_count = 0
        
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            try:
                if os.path.getmtime(file_path) < cutoff_time:
                    os.remove(file_path)
                    removed_count += 1
            except OSError:
                continue
        
        return removed_count


# Global job tracker instance
job_tracker = AsyncJobTracker()