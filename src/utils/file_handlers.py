"""
File processing utilities for handling various file operations.
"""

import os
import shutil
import tempfile
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, BinaryIO, Tuple
import hashlib
from datetime import datetime

from ..utils.logging import LoggingMixin, log_function_call
from ..core.exceptions import FileProcessingError


class FileHandler(LoggingMixin):
    """Utility class for file operations and management."""
    
    # Maximum file sizes (in bytes)
    MAX_FILE_SIZES = {
        'text/plain': 10 * 1024 * 1024,  # 10MB
        'application/pdf': 50 * 1024 * 1024,  # 50MB
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 50 * 1024 * 1024,  # 50MB
        'image/jpeg': 20 * 1024 * 1024,  # 20MB
        'image/png': 20 * 1024 * 1024,  # 20MB
        'audio/wav': 100 * 1024 * 1024,  # 100MB
        'video/mp4': 200 * 1024 * 1024,  # 200MB
    }
    
    # Safe file extensions
    SAFE_EXTENSIONS = {
        '.txt', '.pdf', '.docx', '.xlsx', '.pptx',
        '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff',
        '.wav', '.mp3', '.m4a', '.mp4', '.avi'
    }
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize file handler.
        
        Args:
            temp_dir: Custom temporary directory path
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self._ensure_temp_dir()
    
    def _ensure_temp_dir(self):
        """Ensure temporary directory exists."""
        try:
            os.makedirs(self.temp_dir, exist_ok=True)
        except Exception as e:
            self.logger.error("Failed to create temporary directory", temp_dir=self.temp_dir, error=str(e))
            raise FileProcessingError(f"Cannot create temporary directory: {str(e)}")
    
    @log_function_call
    def validate_file(
        self,
        filename: str,
        file_size: int,
        content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate file properties before processing.
        
        Args:
            filename: Original filename
            file_size: File size in bytes
            content_type: MIME content type
            
        Returns:
            Validation results
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "file_info": {
                "filename": filename,
                "file_size": file_size,
                "content_type": content_type
            }
        }
        
        # Validate filename
        if not filename or not filename.strip():
            validation_result["valid"] = False
            validation_result["errors"].append("Filename is required")
            return validation_result
        
        # Check for dangerous characters
        dangerous_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
        if any(char in filename for char in dangerous_chars):
            validation_result["valid"] = False
            validation_result["errors"].append("Filename contains dangerous characters")
        
        # Validate file extension
        file_extension = Path(filename).suffix.lower()
        if file_extension not in self.SAFE_EXTENSIONS:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Unsupported file extension: {file_extension}")
        
        # Detect MIME type
        detected_mime, _ = mimetypes.guess_type(filename)
        if content_type and detected_mime and content_type != detected_mime:
            validation_result["warnings"].append(
                f"Content type mismatch: provided '{content_type}', detected '{detected_mime}'"
            )
        
        # Use detected MIME type if not provided
        effective_mime = content_type or detected_mime
        validation_result["file_info"]["effective_mime_type"] = effective_mime
        
        # Validate file size
        if file_size <= 0:
            validation_result["valid"] = False
            validation_result["errors"].append("File size must be greater than 0")
        
        # Check maximum file size for type
        if effective_mime in self.MAX_FILE_SIZES:
            max_size = self.MAX_FILE_SIZES[effective_mime]
            if file_size > max_size:
                validation_result["valid"] = False
                validation_result["errors"].append(
                    f"File size ({file_size} bytes) exceeds maximum for type {effective_mime} ({max_size} bytes)"
                )
        else:
            # Default maximum size
            default_max = 50 * 1024 * 1024  # 50MB
            if file_size > default_max:
                validation_result["valid"] = False
                validation_result["errors"].append(
                    f"File size ({file_size} bytes) exceeds default maximum ({default_max} bytes)"
                )
        
        return validation_result
    
    @log_function_call
    def generate_safe_filename(
        self,
        original_filename: str,
        prefix: Optional[str] = None,
        timestamp: bool = True
    ) -> str:
        """
        Generate a safe filename for storage.
        
        Args:
            original_filename: Original filename
            prefix: Optional prefix
            timestamp: Whether to include timestamp
            
        Returns:
            Safe filename
        """
        # Clean the filename
        base_name = Path(original_filename).stem
        extension = Path(original_filename).suffix.lower()
        
        # Remove dangerous characters
        safe_base = "".join(c for c in base_name if c.isalnum() or c in ('-', '_', '.')).strip()
        
        # Ensure it's not empty
        if not safe_base:
            safe_base = "file"
        
        # Add components
        components = []
        if prefix:
            components.append(prefix)
        components.append(safe_base)
        
        if timestamp:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            components.append(timestamp_str)
        
        # Combine and add extension
        safe_filename = "_".join(components) + extension
        
        return safe_filename
    
    @log_function_call
    def calculate_file_hash(self, file_data: bytes, algorithm: str = "sha256") -> str:
        """
        Calculate hash of file content.
        
        Args:
            file_data: File content as bytes
            algorithm: Hash algorithm to use
            
        Returns:
            Hexadecimal hash string
        """
        try:
            if algorithm == "sha256":
                hasher = hashlib.sha256()
            elif algorithm == "md5":
                hasher = hashlib.md5()
            elif algorithm == "sha1":
                hasher = hashlib.sha1()
            else:
                raise ValueError(f"Unsupported hash algorithm: {algorithm}")
            
            hasher.update(file_data)
            return hasher.hexdigest()
            
        except Exception as e:
            self.logger.error("Failed to calculate file hash", algorithm=algorithm, error=str(e))
            raise FileProcessingError(f"Hash calculation failed: {str(e)}")
    
    @log_function_call
    def create_temp_file(
        self,
        file_data: bytes,
        suffix: Optional[str] = None,
        prefix: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Create a temporary file with the given data.
        
        Args:
            file_data: File content as bytes
            suffix: File extension (e.g., '.pdf')
            prefix: Filename prefix
            
        Returns:
            Tuple of (file_path, file_handle_name)
        """
        try:
            # Create temporary file
            fd, temp_path = tempfile.mkstemp(
                suffix=suffix,
                prefix=prefix,
                dir=self.temp_dir
            )
            
            # Write data and close file descriptor
            try:
                os.write(fd, file_data)
            finally:
                os.close(fd)
            
            self.logger.debug("Created temporary file", temp_path=temp_path, size=len(file_data))
            return temp_path, os.path.basename(temp_path)
            
        except Exception as e:
            self.logger.error("Failed to create temporary file", error=str(e))
            raise FileProcessingError(f"Temporary file creation failed: {str(e)}")
    
    @log_function_call
    def cleanup_temp_file(self, file_path: str) -> bool:
        """
        Clean up a temporary file.
        
        Args:
            file_path: Path to temporary file
            
        Returns:
            True if successful
        """
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                self.logger.debug("Cleaned up temporary file", file_path=file_path)
                return True
            return False
            
        except Exception as e:
            self.logger.warning("Failed to cleanup temporary file", file_path=file_path, error=str(e))
            return False
    
    @log_function_call
    def extract_file_metadata(
        self,
        file_data: bytes,
        filename: str,
        content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract metadata from file.
        
        Args:
            file_data: File content as bytes
            filename: Original filename
            content_type: MIME content type
            
        Returns:
            File metadata
        """
        metadata = {
            "filename": filename,
            "file_size": len(file_data),
            "content_type": content_type,
            "file_hash": self.calculate_file_hash(file_data),
            "extracted_at": datetime.utcnow().isoformat()
        }
        
        # Add file extension info
        path_obj = Path(filename)
        metadata.update({
            "file_extension": path_obj.suffix.lower(),
            "base_name": path_obj.stem,
            "safe_filename": self.generate_safe_filename(filename)
        })
        
        # Detect MIME type if not provided
        if not content_type:
            detected_mime, _ = mimetypes.guess_type(filename)
            metadata["detected_mime_type"] = detected_mime
        
        return metadata
    
    @log_function_call
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about a file on disk.
        
        Args:
            file_path: Path to file
            
        Returns:
            File information
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            stat = os.stat(file_path)
            path_obj = Path(file_path)
            
            return {
                "file_path": file_path,
                "filename": path_obj.name,
                "file_size": stat.st_size,
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "accessed_at": datetime.fromtimestamp(stat.st_atime).isoformat(),
                "file_extension": path_obj.suffix.lower(),
                "is_file": path_obj.is_file(),
                "is_directory": path_obj.is_dir(),
                "permissions": oct(stat.st_mode)[-3:]
            }
            
        except Exception as e:
            self.logger.error("Failed to get file info", file_path=file_path, error=str(e))
            raise FileProcessingError(f"Cannot get file info: {str(e)}")
    
    @log_function_call
    def copy_file(self, source_path: str, destination_path: str) -> bool:
        """
        Copy file from source to destination.
        
        Args:
            source_path: Source file path
            destination_path: Destination file path
            
        Returns:
            True if successful
        """
        try:
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            
            # Copy file
            shutil.copy2(source_path, destination_path)
            
            self.logger.debug("File copied successfully", source=source_path, destination=destination_path)
            return True
            
        except Exception as e:
            self.logger.error("Failed to copy file", source=source_path, destination=destination_path, error=str(e))
            raise FileProcessingError(f"File copy failed: {str(e)}")
    
    @log_function_call
    def move_file(self, source_path: str, destination_path: str) -> bool:
        """
        Move file from source to destination.
        
        Args:
            source_path: Source file path
            destination_path: Destination file path
            
        Returns:
            True if successful
        """
        try:
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            
            # Move file
            shutil.move(source_path, destination_path)
            
            self.logger.debug("File moved successfully", source=source_path, destination=destination_path)
            return True
            
        except Exception as e:
            self.logger.error("Failed to move file", source=source_path, destination=destination_path, error=str(e))
            raise FileProcessingError(f"File move failed: {str(e)}")
    
    @log_function_call
    def cleanup_old_temp_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up old temporary files.
        
        Args:
            max_age_hours: Maximum age of files to keep (in hours)
            
        Returns:
            Number of files cleaned up
        """
        try:
            current_time = datetime.now().timestamp()
            max_age_seconds = max_age_hours * 3600
            cleaned_count = 0
            
            for file_path in Path(self.temp_dir).glob("*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        try:
                            file_path.unlink()
                            cleaned_count += 1
                        except Exception as e:
                            self.logger.warning("Failed to cleanup old temp file", file_path=str(file_path), error=str(e))
            
            self.logger.info("Cleaned up old temporary files", count=cleaned_count, max_age_hours=max_age_hours)
            return cleaned_count
            
        except Exception as e:
            self.logger.error("Failed to cleanup old temp files", error=str(e))
            return 0
    
    def __del__(self):
        """Cleanup on destruction."""
        # Clean up any remaining temporary files
        try:
            self.cleanup_old_temp_files(max_age_hours=1)  # Clean files older than 1 hour
        except Exception:
            pass  # Ignore errors during cleanup


# Utility functions for common file operations
def validate_upload_file(filename: str, file_size: int, content_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to validate an upload file.
    
    Args:
        filename: Original filename
        file_size: File size in bytes
        content_type: MIME content type
        
    Returns:
        Validation results
    """
    handler = FileHandler()
    return handler.validate_file(filename, file_size, content_type)


def create_temp_file_from_upload(file_data: bytes, filename: str) -> Tuple[str, str]:
    """
    Convenience function to create temporary file from upload data.
    
    Args:
        file_data: File content as bytes
        filename: Original filename
        
    Returns:
        Tuple of (file_path, temp_filename)
    """
    handler = FileHandler()
    suffix = Path(filename).suffix
    return handler.create_temp_file(file_data, suffix=suffix)


def cleanup_temp_file(file_path: str) -> bool:
    """
    Convenience function to cleanup temporary file.
    
    Args:
        file_path: Path to temporary file
        
    Returns:
        True if successful
    """
    handler = FileHandler()
    return handler.cleanup_temp_file(file_path)