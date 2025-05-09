"""API router for the CroceRossa Qdrant Cloud application."""

from fastapi import APIRouter, Depends, HTTPException, Body
from typing import Dict, Any, Optional

from app.api.models import (
    QueryRequest,
    QueryResponse,
    ResetRequest,
    ResetResponse,
    TranscriptResponse,
    ContactResponse,
)
from app.core.config import settings
from app.core.logging import get_logger
from app.rag.engine import RAGEngine
from app.rag.memory import ConversationMemory

logger = get_logger(__name__)

router = APIRouter()

# Global store for session-specific memories
session_memories: Dict[str, ConversationMemory] = {}

def get_session_memory(session_id: Optional[str] = None) -> ConversationMemory:
    """Dependency to get or create session-specific conversation memory."""
    if not session_id:
        logger.warning("Request received without session_id, creating temporary memory")
        return ConversationMemory()

    if session_id not in session_memories:
        logger.info(f"Creating new conversation memory for session_id: {session_id}")
        session_memories[session_id] = ConversationMemory()
        
    memory = session_memories[session_id]
    logger.debug(f"Retrieved memory for session_id: {session_id} (memory ID: {id(memory)})")
    return memory

# Dependency to get RAG engine, now aware of session-specific memory
def get_rag_engine(session_memory: ConversationMemory = Depends(get_session_memory)) -> RAGEngine:
    """Dependency to provide the RAG engine instance, initialized with session-specific memory."""
    try:
        # Pass the session-specific memory to the RAGEngine
        engine = RAGEngine(memory=session_memory)
        logger.debug(f"Created RAG engine with memory ID: {id(session_memory)}")
        
        # Check for the internal failure flag set during RAGEngine init
        if hasattr(engine, '_initialization_failed') and engine._initialization_failed:
            logger.error("RAGEngine initialization failed (detected in get_rag_engine via flag)")
            # We can't raise HTTPException here directly as it's a dependency setup
        return engine
    except Exception as e:
        logger.error(f"Failed to initialize RAG engine in dependency: {str(e)}", exc_info=True)
        # Create a minimally functional engine for error reporting
        engine = RAGEngine.__new__(RAGEngine) # Create instance without calling __init__
        setattr(engine, '_initialization_failed', True) # Manually set failure flag
        setattr(engine, 'memory', session_memory) # Assign the session memory
        logger.warning("Returning a minimally functional RAGEngine due to init error.")
        return engine


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, rag_engine: RAGEngine = Depends(get_rag_engine)):
    """Process a user query and return a response."""
    logger.info(f"Received query: '{request.query}', session_id: {request.session_id}")
    
    # Get the memory specific to this session
    current_session_memory = get_session_memory(request.session_id)
    
    # Log the memory state before loading history
    logger.info(f"Memory before loading history: {len(current_session_memory.get_history())} exchanges")
    
    # Load conversation history
    if request.conversation_history:
        logger.info(f"Loading {len(request.conversation_history)} items from client history")
        current_session_memory.load_history(request.conversation_history)
        logger.info(f"Memory after loading: {len(current_session_memory.get_history())} exchanges")
    
    # Make sure the RAG engine uses this memory
    rag_engine.memory = current_session_memory
    
    # Process the query
    try:
        result = rag_engine.query(request.query, include_prompt=request.include_prompt)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Si è verificato un errore durante l'elaborazione della richiesta: {str(e)}"
        )


@router.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest):
    """Reset the conversation memory for a given session_id."""
    logger.info(f"Received reset request for session_id: {request.session_id}")
    
    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id is required for reset")

    try:
        # Get the memory for this session
        if request.session_id in session_memories:
            # Reset the memory
            session_memories[request.session_id].reset()
            # Also remove from global store for a complete reset
            del session_memories[request.session_id]
            logger.info(f"Reset and removed memory for session_id: {request.session_id}")
        else:
            logger.warning(f"Reset requested for non-existent session_id: {request.session_id}")
        
        return ResetResponse(
            success=True,
            message="Conversazione resettata con successo."
        )
    
    except Exception as e:
        logger.error(f"Error resetting conversation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Si è verificato un errore durante il reset della conversazione: {str(e)}"
        )


@router.get("/transcript", response_model=TranscriptResponse)
async def transcript(session_id: Optional[str] = None):
    """Get the conversation transcript for a given session_id."""
    logger.info(f"Received transcript request for session_id: {session_id}")
    
    # Il frontend sta chiamando questo endpoint senza session_id
    # Prova a usare i cookie o l'ultimo session_id attivo
    if not session_id and session_memories:
        # Usa l'ultimo session_id attivo come fallback
        active_sessions = list(session_memories.keys())
        if active_sessions:
            session_id = active_sessions[-1]
            logger.info(f"No session_id provided, using last active session: {session_id}")

    if not session_id:
        logger.warning("No session_id provided and no active sessions found")
        return TranscriptResponse(transcript=[])

    try:
        if session_id in session_memories:
            transcript_data = session_memories[session_id].get_transcript()
            logger.info(f"Returning transcript with {len(transcript_data)} exchanges for session_id: {session_id}")
            return TranscriptResponse(transcript=transcript_data)
        else:
            logger.warning(f"Transcript requested for non-existent session_id: {session_id}")
            return TranscriptResponse(transcript=[])
    
    except Exception as e:
        logger.error(f"Error retrieving transcript: {str(e)}", exc_info=True)
        return TranscriptResponse(transcript=[])


@router.get("/contact", response_model=ContactResponse)
async def contact():
    """Get the CRI contact information."""
    logger.info("Received contact information request")
    
    try:
        return ContactResponse(
            name="Croce Rossa Italiana",
            website=settings.CRI_WEBSITE,
            email=settings.CRI_CONTACT_EMAIL,
            phone=settings.CRI_CONTACT_PHONE,
            headquarters="Via Bernardino Ramazzini, 31, 00151 Roma RM",
            description=(
                "La Croce Rossa Italiana, fondata il 15 giugno 1864, è un'associazione "
                "di soccorso volontario, parte integrante del Movimento Internazionale "
                "della Croce Rossa e della Mezzaluna Rossa. Opera in Italia nei campi "
                "sanitario, sociale e umanitario, secondo i sette Principi Fondamentali "
                "del Movimento: Umanità, Imparzialità, Neutralità, Indipendenza, "
                "Volontariato, Unità e Universalità."
            )
        )
    
    except Exception as e:
        logger.error(f"Error retrieving contact information: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Si è verificato un errore durante il recupero delle informazioni di contatto: {str(e)}"
        )