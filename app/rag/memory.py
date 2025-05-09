"""Conversation memory management for the CroceRossa Qdrant Cloud application."""

import os
import sys
import json
from typing import Dict, List, Tuple, Optional
from collections import deque

# Add project root to Python path when running this file directly
if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.insert(0, PROJECT_ROOT)
    print(f"Added project root to Python path: {PROJECT_ROOT}")

from app.core.config import settings # settings è già istanziato in config.py
from app.core.logging import get_logger

logger = get_logger(__name__)


class ConversationMemory:
    """Manages conversation history with a fixed window size.
    
    Stores pairs of user questions and system responses in a sliding window,
    allowing for context-aware follow-up question handling.
    """
    
    def __init__(self, window_size: Optional[int] = None):
        """Initialize the conversation memory.
        
        Args:
            window_size: Maximum number of exchanges to keep in memory.
                         Defaults to MEMORY_WINDOW_SIZE from settings.
        """
        self.window_size = window_size if window_size is not None else settings.MEMORY_WINDOW_SIZE
        self.memory: deque[Tuple[str, str]] = deque(maxlen=self.window_size)
        self.transcript: List[Dict[str, str]] = [] # Full transcript, not windowed
        logger.info(f"Initialized conversation memory with window size {self.window_size}")
        
    def add_exchange(self, question: str, answer: str) -> None:
        """Add a question-answer exchange to the memory.
        
        Args:
            question: The user's question
            answer: The system's response
        """
        if not question or not answer: # Basic validation
            logger.warning("Attempted to add empty question or answer to memory. Skipping.")
            return
            
        self.memory.append((question, answer))
        self.transcript.append({"user": question, "assistant": answer})
        logger.info(f"Added exchange to memory. Current memory (deque) size: {len(self.memory)}")
        
    def get_history(self) -> List[Tuple[str, str]]:
        """Get the current conversation history from the sliding window.
        
        Returns:
            List of (question, answer) tuples from the conversation window.
        """
        history = list(self.memory)
        logger.debug(f"Retrieved {len(history)} exchanges from memory window.")
        return history
    
    def get_transcript(self) -> List[Dict[str, str]]:
        """Get the full conversation transcript (not windowed).
        
        Returns:
            List of dictionaries with user questions and assistant answers.
        """
        logger.debug(f"Retrieved full transcript with {len(self.transcript)} exchanges.")
        return list(self.transcript) # Return a copy
    
    def reset(self) -> None:
        """Reset the conversation memory and transcript."""
        self.memory.clear()
        self.transcript.clear() # Clear the full transcript as well
        logger.info("Conversation memory (window and transcript) reset.")

    def load_history(self, history_items: List[Dict[str, str]]) -> None:
        """Load conversation history from a list of message dictionaries.
        This will clear existing memory and transcript before loading.
        
        Args:
            history_items: List of dictionaries, each should have "type" ('user' or 'assistant')
                           and "content" (the message text).
                           Example: [{"type": "user", "content": "Ciao"}, {"type": "assistant", "content": "Salve!"}]
        """
        if not history_items:
            logger.warning("Empty history_items provided to load_history. No action taken.")
            return
            
        logger.info(f"Loading history with {len(history_items)} items into memory.")
        
        self.reset() # Clear current memory and transcript first
        
        # Process history items to extract user-assistant pairs
        user_q: Optional[str] = None
        for item in history_items:
            msg_type = item.get("type")
            content = item.get("content")

            if not msg_type or not content:
                logger.warning(f"Skipping history item with missing type or content: {item}")
                continue

            if msg_type == "user":
                if user_q is not None: # Found a user message without a preceding assistant; log and overwrite
                    logger.warning(f"Found consecutive user messages in history. Overwriting previous user query: '{user_q}'")
                user_q = content
            elif msg_type == "assistant":
                if user_q is not None:
                    self.add_exchange(user_q, content)
                    logger.debug(f"Loaded exchange from history: User='{user_q[:30]}...', Assistant='{content[:30]}...'")
                    user_q = None # Reset user_q after pairing
                else:
                    # Found an assistant message without a preceding user message
                    logger.warning(f"Found assistant message without preceding user message in history. Skipping: '{content[:50]}...'")
            else:
                logger.warning(f"Unknown message type '{msg_type}' in history item. Skipping: {item}")
        
        if user_q is not None: # Unpaired user message at the end
            logger.warning(f"Unpaired user message at the end of history load: '{user_q}'. Not added to memory exchanges.")

        logger.info(f"Successfully loaded {len(self.memory)} exchanges into memory window from {len(history_items)} items.")

    def is_follow_up_question(self) -> bool:
        """Check if there's any conversation history in the window.
        
        Returns:
            True if there is conversation history, False otherwise.
        """
        has_history = len(self.memory) > 0
        logger.debug(f"Is follow-up question check: {has_history} (memory window size: {len(self.memory)})")
        return has_history