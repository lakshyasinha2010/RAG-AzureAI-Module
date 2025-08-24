"""
Input validation utilities for the RAG service.
"""

import re
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from urllib.parse import urlparse
from email.utils import parseaddr

from ..utils.logging import LoggingMixin, log_function_call
from ..core.exceptions import ValidationError
from ..models.schemas import SearchType, ContentType


class InputValidator(LoggingMixin):
    """Utility class for input validation and sanitization."""
    
    # Regular expressions for validation
    PATTERNS = {
        'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
        'alphanumeric': re.compile(r'^[a-zA-Z0-9]+$'),
        'safe_string': re.compile(r'^[a-zA-Z0-9\s\-_.]+$'),
        'source_id': re.compile(r'^[a-zA-Z0-9\-_]{1,50}$'),
        'semantic_version': re.compile(r'^\d+\.\d+\.\d+$'),
        'uuid': re.compile(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'),
    }
    
    # Safe characters for different contexts
    SAFE_CHARS = {
        'filename': set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.'),
        'query': set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_.?!,;:()[]{}'),
        'identifier': set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_'),
    }
    
    def __init__(self):
        """Initialize the input validator."""
        pass
    
    @log_function_call
    def validate_query_text(
        self,
        query: str,
        min_length: int = 1,
        max_length: int = 2000,
        allow_empty: bool = False
    ) -> Dict[str, Any]:
        """
        Validate query text input.
        
        Args:
            query: Query text to validate
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            allow_empty: Whether to allow empty queries
            
        Returns:
            Validation results
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "sanitized_query": query
        }
        
        # Check for None or non-string input
        if query is None:
            if allow_empty:
                result["sanitized_query"] = ""
                return result
            else:
                result["valid"] = False
                result["errors"].append("Query cannot be None")
                return result
        
        if not isinstance(query, str):
            result["valid"] = False
            result["errors"].append("Query must be a string")
            return result
        
        # Sanitize query
        sanitized = query.strip()
        result["sanitized_query"] = sanitized
        
        # Check length
        if not allow_empty and len(sanitized) == 0:
            result["valid"] = False
            result["errors"].append("Query cannot be empty")
        elif len(sanitized) < min_length:
            result["valid"] = False
            result["errors"].append(f"Query too short (minimum {min_length} characters)")
        elif len(sanitized) > max_length:
            result["valid"] = False
            result["errors"].append(f"Query too long (maximum {max_length} characters)")
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'<script.*?</script>',  # Script tags
            r'javascript:',          # JavaScript protocol
            r'on\w+\s*=',           # Event handlers
            r'eval\s*\(',           # Eval function
            r'document\.',          # DOM access
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                result["valid"] = False
                result["errors"].append("Query contains potentially dangerous content")
                break
        
        # Check for excessive whitespace
        if len(sanitized) != len(' '.join(sanitized.split())):
            result["warnings"].append("Query contains excessive whitespace")
            result["sanitized_query"] = ' '.join(sanitized.split())
        
        return result
    
    @log_function_call
    def validate_filename(
        self,
        filename: str,
        max_length: int = 255,
        allow_unicode: bool = False
    ) -> Dict[str, Any]:
        """
        Validate filename input.
        
        Args:
            filename: Filename to validate
            max_length: Maximum allowed length
            allow_unicode: Whether to allow Unicode characters
            
        Returns:
            Validation results
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "sanitized_filename": filename
        }
        
        if not filename or not isinstance(filename, str):
            result["valid"] = False
            result["errors"].append("Filename must be a non-empty string")
            return result
        
        # Basic sanitization
        sanitized = filename.strip()
        
        # Check length
        if len(sanitized) == 0:
            result["valid"] = False
            result["errors"].append("Filename cannot be empty")
        elif len(sanitized) > max_length:
            result["valid"] = False
            result["errors"].append(f"Filename too long (maximum {max_length} characters)")
        
        # Check for dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '/', '\\']
        for char in dangerous_chars:
            if char in sanitized:
                result["valid"] = False
                result["errors"].append(f"Filename contains dangerous character: {char}")
        
        # Check for reserved names (Windows)
        reserved_names = ['CON', 'PRN', 'AUX', 'NUL'] + [f'COM{i}' for i in range(1, 10)] + [f'LPT{i}' for i in range(1, 10)]
        if sanitized.upper() in reserved_names:
            result["valid"] = False
            result["errors"].append("Filename is a reserved system name")
        
        # Check for Unicode characters if not allowed
        if not allow_unicode and not sanitized.isascii():
            result["valid"] = False
            result["errors"].append("Filename contains non-ASCII characters")
        
        # Check for leading/trailing dots or spaces
        if sanitized.startswith('.') or sanitized.endswith('.'):
            result["warnings"].append("Filename starts or ends with a dot")
        if sanitized.startswith(' ') or sanitized.endswith(' '):
            result["warnings"].append("Filename starts or ends with a space")
            sanitized = sanitized.strip()
        
        result["sanitized_filename"] = sanitized
        return result
    
    @log_function_call
    def validate_source_id(self, source_id: str) -> Dict[str, Any]:
        """
        Validate source ID format.
        
        Args:
            source_id: Source ID to validate
            
        Returns:
            Validation results
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "sanitized_source_id": source_id
        }
        
        if not source_id or not isinstance(source_id, str):
            result["valid"] = False
            result["errors"].append("Source ID must be a non-empty string")
            return result
        
        sanitized = source_id.strip()
        
        if not self.PATTERNS['source_id'].match(sanitized):
            result["valid"] = False
            result["errors"].append("Source ID must contain only alphanumeric characters, hyphens, and underscores (1-50 chars)")
        
        result["sanitized_source_id"] = sanitized
        return result
    
    @log_function_call
    def validate_search_type(self, search_type: Union[str, SearchType]) -> Dict[str, Any]:
        """
        Validate search type.
        
        Args:
            search_type: Search type to validate
            
        Returns:
            Validation results
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "search_type": None
        }
        
        if isinstance(search_type, SearchType):
            result["search_type"] = search_type
            return result
        
        if isinstance(search_type, str):
            try:
                result["search_type"] = SearchType(search_type.lower())
                return result
            except ValueError:
                result["valid"] = False
                result["errors"].append(f"Invalid search type: {search_type}. Must be one of: {[t.value for t in SearchType]}")
        else:
            result["valid"] = False
            result["errors"].append("Search type must be a string or SearchType enum")
        
        return result
    
    @log_function_call
    def validate_content_type(self, content_type: Union[str, ContentType]) -> Dict[str, Any]:
        """
        Validate content type.
        
        Args:
            content_type: Content type to validate
            
        Returns:
            Validation results
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "content_type": None
        }
        
        if isinstance(content_type, ContentType):
            result["content_type"] = content_type
            return result
        
        if isinstance(content_type, str):
            try:
                result["content_type"] = ContentType(content_type.lower())
                return result
            except ValueError:
                result["valid"] = False
                result["errors"].append(f"Invalid content type: {content_type}. Must be one of: {[t.value for t in ContentType]}")
        else:
            result["valid"] = False
            result["errors"].append("Content type must be a string or ContentType enum")
        
        return result
    
    @log_function_call
    def validate_metadata(self, metadata: Any) -> Dict[str, Any]:
        """
        Validate metadata input.
        
        Args:
            metadata: Metadata to validate
            
        Returns:
            Validation results
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "sanitized_metadata": metadata
        }
        
        if metadata is None:
            result["sanitized_metadata"] = {}
            return result
        
        # If it's a string, try to parse as JSON
        if isinstance(metadata, str):
            try:
                parsed_metadata = json.loads(metadata)
                result["sanitized_metadata"] = parsed_metadata
                metadata = parsed_metadata
            except json.JSONDecodeError as e:
                result["valid"] = False
                result["errors"].append(f"Invalid JSON format in metadata: {str(e)}")
                return result
        
        # Must be a dictionary
        if not isinstance(metadata, dict):
            result["valid"] = False
            result["errors"].append("Metadata must be a dictionary or valid JSON string")
            return result
        
        # Validate metadata size
        try:
            metadata_json = json.dumps(metadata)
            if len(metadata_json) > 10000:  # 10KB limit
                result["valid"] = False
                result["errors"].append("Metadata too large (maximum 10KB when serialized)")
        except (TypeError, ValueError) as e:
            result["valid"] = False
            result["errors"].append(f"Metadata cannot be serialized to JSON: {str(e)}")
        
        # Check for dangerous keys or values
        dangerous_keys = ['__proto__', 'constructor', 'prototype']
        for key in metadata.keys():
            if key in dangerous_keys:
                result["valid"] = False
                result["errors"].append(f"Metadata contains dangerous key: {key}")
            if not isinstance(key, str):
                result["valid"] = False
                result["errors"].append("All metadata keys must be strings")
        
        return result
    
    @log_function_call
    def validate_numeric_range(
        self,
        value: Union[int, float],
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        value_name: str = "value"
    ) -> Dict[str, Any]:
        """
        Validate numeric value within a range.
        
        Args:
            value: Numeric value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            value_name: Name of the value for error messages
            
        Returns:
            Validation results
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "value": value
        }
        
        if not isinstance(value, (int, float)):
            result["valid"] = False
            result["errors"].append(f"{value_name} must be a number")
            return result
        
        if min_value is not None and value < min_value:
            result["valid"] = False
            result["errors"].append(f"{value_name} must be at least {min_value}")
        
        if max_value is not None and value > max_value:
            result["valid"] = False
            result["errors"].append(f"{value_name} must be at most {max_value}")
        
        return result
    
    @log_function_call
    def validate_odata_filter(self, filter_string: Optional[str]) -> Dict[str, Any]:
        """
        Validate OData filter string format.
        
        Args:
            filter_string: OData filter string
            
        Returns:
            Validation results
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "sanitized_filter": filter_string
        }
        
        if not filter_string:
            return result
        
        if not isinstance(filter_string, str):
            result["valid"] = False
            result["errors"].append("Filter must be a string")
            return result
        
        sanitized = filter_string.strip()
        result["sanitized_filter"] = sanitized
        
        # Basic validation for OData filter syntax
        # Check for dangerous patterns
        dangerous_patterns = [
            r'DROP\s+TABLE',
            r'DELETE\s+FROM',
            r'INSERT\s+INTO',
            r'UPDATE\s+SET',
            r'EXEC\s*\(',
            r'UNION\s+SELECT',
            r'<script',
            r'javascript:',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                result["valid"] = False
                result["errors"].append("Filter contains potentially dangerous content")
                break
        
        # Check filter length
        if len(sanitized) > 1000:
            result["valid"] = False
            result["errors"].append("Filter string too long (maximum 1000 characters)")
        
        return result
    
    @log_function_call
    def validate_email(self, email: str) -> Dict[str, Any]:
        """
        Validate email address format.
        
        Args:
            email: Email address to validate
            
        Returns:
            Validation results
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "sanitized_email": email
        }
        
        if not email or not isinstance(email, str):
            result["valid"] = False
            result["errors"].append("Email must be a non-empty string")
            return result
        
        sanitized = email.strip().lower()
        result["sanitized_email"] = sanitized
        
        # Basic format validation
        if not self.PATTERNS['email'].match(sanitized):
            result["valid"] = False
            result["errors"].append("Invalid email format")
        
        # Additional validation using email.utils
        try:
            parsed_name, parsed_email = parseaddr(sanitized)
            if not parsed_email or '@' not in parsed_email:
                result["valid"] = False
                result["errors"].append("Invalid email format")
        except Exception:
            result["valid"] = False
            result["errors"].append("Invalid email format")
        
        return result
    
    @log_function_call
    def validate_url(self, url: str) -> Dict[str, Any]:
        """
        Validate URL format.
        
        Args:
            url: URL to validate
            
        Returns:
            Validation results
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "sanitized_url": url
        }
        
        if not url or not isinstance(url, str):
            result["valid"] = False
            result["errors"].append("URL must be a non-empty string")
            return result
        
        sanitized = url.strip()
        result["sanitized_url"] = sanitized
        
        try:
            parsed = urlparse(sanitized)
            
            # Check for required components
            if not parsed.scheme:
                result["valid"] = False
                result["errors"].append("URL must have a scheme (http, https, etc.)")
            elif parsed.scheme not in ['http', 'https', 'ftp', 'ftps']:
                result["warnings"].append(f"Unusual URL scheme: {parsed.scheme}")
            
            if not parsed.netloc:
                result["valid"] = False
                result["errors"].append("URL must have a network location (domain)")
            
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Invalid URL format: {str(e)}")
        
        return result
    
    @log_function_call
    def sanitize_text_input(
        self,
        text: str,
        max_length: Optional[int] = None,
        allowed_chars: Optional[str] = None
    ) -> str:
        """
        Sanitize text input by removing dangerous characters.
        
        Args:
            text: Text to sanitize
            max_length: Maximum allowed length
            allowed_chars: Set of allowed characters (default: safe_chars['query'])
            
        Returns:
            Sanitized text
        """
        if not isinstance(text, str):
            return ""
        
        # Use default allowed characters if not specified
        if allowed_chars is None:
            allowed_chars = self.SAFE_CHARS['query']
        
        # Remove disallowed characters
        sanitized = ''.join(char for char in text if char in allowed_chars)
        
        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        # Truncate if necessary
        if max_length and len(sanitized) > max_length:
            sanitized = sanitized[:max_length].strip()
        
        return sanitized


# Convenience functions
def validate_query(query: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to validate query text."""
    validator = InputValidator()
    return validator.validate_query_text(query, **kwargs)


def validate_filename(filename: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to validate filename."""
    validator = InputValidator()
    return validator.validate_filename(filename, **kwargs)


def sanitize_text(text: str, **kwargs) -> str:
    """Convenience function to sanitize text input."""
    validator = InputValidator()
    return validator.sanitize_text_input(text, **kwargs)


def validate_range(value: Union[int, float], min_val: Optional[Union[int, float]] = None, 
                  max_val: Optional[Union[int, float]] = None, name: str = "value") -> Dict[str, Any]:
    """Convenience function to validate numeric range."""
    validator = InputValidator()
    return validator.validate_numeric_range(value, min_val, max_val, name)