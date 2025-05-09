"""Utility functions for the CroceRossa Qdrant Cloud application."""

import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from app.core.logging import get_logger

logger = get_logger(__name__)


def safe_json_serialize(obj: Any) -> Dict[str, Any]:
    """Safely serialize an object to JSON, handling non-serializable types.
    
    Args:
        obj: The object to serialize
        
    Returns:
        A JSON-serializable dictionary
    """
    try:
        # Try to directly serialize
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        # If direct serialization fails, convert to a safe representation
        if isinstance(obj, dict):
            return {k: safe_json_serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [safe_json_serialize(item) for item in obj]
        elif isinstance(obj, (datetime,)):
            return obj.isoformat()
        elif hasattr(obj, "__dict__"):
            return safe_json_serialize(obj.__dict__)
        else:
            return str(obj)


def format_conversation_for_export(transcript: List[Dict[str, str]], 
                                  include_timestamp: bool = True) -> Dict[str, Any]:
    """Format conversation transcript for export.
    
    Args:
        transcript: The conversation transcript
        include_timestamp: Whether to include timestamp in the export
        
    Returns:
        A formatted dictionary for export
    """
    result = {
        "conversation": transcript,
        "metadata": {
            "count": len(transcript),
        }
    }
    
    if include_timestamp:
        result["metadata"]["exported_at"] = datetime.now().isoformat()
    
    return result


def save_conversation_to_file(transcript: List[Dict[str, str]], 
                             filepath: str,
                             overwrite: bool = False) -> bool:
    """Save conversation transcript to a file.
    
    Args:
        transcript: The conversation transcript
        filepath: Path to the output file
        overwrite: Whether to overwrite existing file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if file exists and overwrite is not allowed
        if os.path.exists(filepath) and not overwrite:
            logger.warning("File exists and overwrite not allowed", filepath=filepath)
            return False
        
        # Format conversation for export
        formatted_transcript = format_conversation_for_export(transcript)
        
        # Write to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(formatted_transcript, f, ensure_ascii=False, indent=2)
        
        logger.info("Conversation saved to file", filepath=filepath)
        return True
    
    except Exception as e:
        logger.error("Error saving conversation to file", error=str(e), filepath=filepath)
        return False