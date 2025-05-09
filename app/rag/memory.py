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

from app.core.config import settings
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
        self.window_size = window_size or settings.MEMORY_WINDOW_SIZE
        self.memory: deque[Tuple[str, str]] = deque(maxlen=self.window_size)
        self.transcript: List[Dict[str, str]] = []
        logger.info(f"Initialized conversation memory with window size {self.window_size}")
        
    def add_exchange(self, question: str, answer: str) -> None:
        """Add a question-answer exchange to the memory.
        
        Args:
            question: The user's question
            answer: The system's response
        """
        if not question or not answer:
            logger.warning("Attempted to add empty question or answer to memory")
            return
            
        self.memory.append((question, answer))
        self.transcript.append({"user": question, "assistant": answer})
        logger.info(f"Added exchange to memory. Current memory size: {len(self.memory)}")
        
        # Log the current state of memory for debugging
        memory_content = [{"question": q, "answer": a[:50] + "..." if len(a) > 50 else a} 
                         for q, a in self.memory]
        logger.debug(f"Current memory state: {json.dumps(memory_content)}")
        
    def get_history(self) -> List[Tuple[str, str]]:
        """Get the current conversation history.
        
        Returns:
            List of (question, answer) tuples from the conversation
        """
        history = list(self.memory)
        logger.debug(f"Retrieved {len(history)} exchanges from memory")
        return history
    
    def get_transcript(self) -> List[Dict[str, str]]:
        """Get the full conversation transcript.
        
        Returns:
            List of dictionaries with user questions and assistant answers
        """
        logger.debug(f"Retrieved transcript with {len(self.transcript)} exchanges")
        return self.transcript
    
    def reset(self) -> None:
        """Reset the conversation memory and transcript."""
        self.memory.clear()
        self.transcript = []
        logger.info("Conversation memory and transcript reset")

    def load_history(self, history_items: List[Dict[str, str]]) -> None:
        """Load conversation history from a list of message dictionaries.
        
        Args:
            history_items: List of dictionaries with 'type' and 'content' keys
        """
        if not history_items:
            logger.warning("Empty history items provided to load_history")
            return
            
        logger.info(f"Loading history with {len(history_items)} items")
        
        # Make a debug log of the incoming history items
        logger.debug(f"History items format: {json.dumps(history_items[:2] if len(history_items) > 2 else history_items)}")
        
        # Clear existing memory and transcript before loading new history
        self.memory.clear()
        self.transcript = []
        
        # Process history items to extract user-assistant pairs
        i = 0
        while i < len(history_items) - 1:  # Stop before the last item to avoid index errors
            # Look for user-assistant pairs
            if (history_items[i].get("type") == "user" and 
                history_items[i+1].get("type") == "assistant"):
                
                user_content = history_items[i].get("content", "")
                assistant_content = history_items[i+1].get("content", "")
                
                if user_content and assistant_content:
                    self.memory.append((user_content, assistant_content))
                    self.transcript.append({
                        "user": user_content,
                        "assistant": assistant_content
                    })
                    logger.debug(f"Added exchange pair from history: Q={user_content[:30]}..., A={assistant_content[:30]}...")
                else:
                    logger.warning(f"Skipped empty content in history items {i} and {i+1}")
                
                # Move to the next potential pair
                i += 2
            else:
                # If not a user-assistant pair, skip one item and look for the next pair
                logger.warning(f"Unexpected message type sequence at position {i}: "
                             f"{history_items[i].get('type')} -> {history_items[i+1].get('type')}")
                i += 1
        
        # Ensure we don't exceed the window size
        while len(self.memory) > self.window_size:
            self.memory.popleft()
            
        logger.info(f"Successfully loaded {len(self.memory)} exchanges into memory")
        
        # Log the current state for debugging
        if self.memory:
            logger.debug(f"Memory now contains {len(self.memory)} exchanges")
            first_exchange = self.memory[0] if self.memory else None
            last_exchange = self.memory[-1] if self.memory else None
            
            if first_exchange:
                logger.debug(f"First exchange: Q={first_exchange[0][:30]}..., A={first_exchange[1][:30]}...")
            if last_exchange:
                logger.debug(f"Last exchange: Q={last_exchange[0][:30]}..., A={last_exchange[1][:30]}...")

    def is_follow_up_question(self) -> bool:
        """Check if there's any conversation history, indicating a follow-up question.
        
        Returns:
            True if there is conversation history, False otherwise
        """
        has_history = len(self.memory) > 0
        logger.debug(f"Checking if follow-up question: {has_history} (memory size: {len(self.memory)})")
        return has_history
        
    def get_recent_history(self, max_exchanges: int = 3) -> List[Tuple[str, str]]:
        """Get the most recent conversation exchanges, up to a specified limit.
        
        Args:
            max_exchanges: Maximum number of recent exchanges to return
            
        Returns:
            List of the most recent (question, answer) tuples
        """
        # Get the last N exchanges from memory
        recent_history = list(self.memory)[-max_exchanges:] if self.memory else []
        logger.debug(f"Retrieved {len(recent_history)} recent exchanges out of {len(self.memory)} total")
        return recent_history