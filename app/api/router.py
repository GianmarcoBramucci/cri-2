"""API router for RAG queries and other application endpoints."""

from fastapi import APIRouter, HTTPException, Depends, Request, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from app.rag.engine import RAGEngine
from app.rag.memory import ConversationMemory 
from app.core.logging import get_logger
from app.core.config import settings as app_settings # Per accedere a settings.CRI_...


logger = get_logger(__name__) # Logger specifico per il router
router = APIRouter()

# --- Pydantic Models for API requests and responses ---
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    # La storia della conversazione ora viene gestita dalla memoria interna dell'engine,
    # ma il client PUO' inviare una storia per "inizializzare" o "sovrascrivere"
    # la memoria per una data sessione (se l'engine supportasse sessioni multiple di memoria).
    # Per ora, RAGEngine ha UNA memoria. Se il client invia la storia, la usiamo per popolare quella.
    conversation_history: Optional[List[Dict[str, str]]] = Field(None, 
        description='Optional: client-provided history. Format: [{"type": "user"|"assistant", "content": "..."}]. If provided, it will replace current engine memory content.')
    include_prompt: Optional[bool] = False

class SourceDocument(BaseModel):
    text_preview: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[SourceDocument]
    condensed_question: Optional[str] = None
    error: Optional[str] = None
    full_prompt: Optional[str] = None # Se include_prompt è True

class ResetRequest(BaseModel): # Non più usato direttamente, ma potrebbe servire per future API specifiche per sessione
    session_id: Optional[str] = None # Opzionale, per resettare la memoria globale dell'engine se non specificato

class TranscriptResponse(BaseModel):
    transcript: List[Dict[str, str]]

class ContactInfo(BaseModel):
    email: str
    phone: str
    website: str


# --- Helper function to get RAG Engine instance ---
def get_rag_engine(request: Request) -> RAGEngine:
    if not hasattr(request.app.state, 'rag_engine') or request.app.state.rag_engine is None:
        logger.error("RAG Engine instance not found in application state. Startup might have failed.")
        raise HTTPException(status_code=503, detail="Servizio RAG temporaneamente non disponibile (Engine non trovato).")
    
    engine: RAGEngine = request.app.state.rag_engine
    if engine._initialization_failed: 
        logger.error("RAG Engine initialization reported failure. Service unavailable.")
        raise HTTPException(status_code=503, detail="Servizio RAG non disponibile: inizializzazione fallita.")
    return engine

# --- API Endpoints ---

@router.post("/query", response_model=QueryResponse)
async def handle_query_endpoint( # Rinomina per evitare conflitto con la funzione `query` dell'engine
    query_request: QueryRequest,
    engine: RAGEngine = Depends(get_rag_engine)
):
    """
    Process a user query using the RAG engine.
    If `conversation_history` is provided by the client, the engine's current memory
    will be reset and loaded with this history before processing the query.
    """
    session_id = query_request.session_id or "default_session" # Usa un default se non fornito

    # Se il client invia una cronologia, la usiamo per popolare la memoria dell'engine per questa chiamata.
    # L'engine attuale ha una singola istanza di memoria.
    # Questo significa che ogni chiamata con `conversation_history` sovrascrive la memoria precedente.
    if query_request.conversation_history is not None: # Permetti lista vuota per resettare
        logger.info(f"Client provided conversation history for session '{session_id}'. Resetting and loading engine memory.")
        try:
            engine.memory.reset() # Assicurati che memory esista
            if query_request.conversation_history: # Carica solo se non è vuota
                 engine.memory.load_history(query_request.conversation_history)
        except AttributeError:
            logger.error("Engine memory object not found while trying to load history. Engine might not be fully initialized.")
            raise HTTPException(status_code=503, detail="Errore interno: gestione memoria fallita.")


    logger.info(f"Processing query via API: '{query_request.query}' for session_id: {session_id}")
    try:
        # Chiama il metodo asincrono aquery dell'engine
        result_dict = await engine.aquery(
            question=query_request.query,
            include_prompt=query_request.include_prompt or False # Default a False se non fornito
        )
        return QueryResponse(**result_dict) # Mappa il dizionario al modello Pydantic
    except Exception as e:
        logger.error(f"Error during engine.aquery for session {session_id}: {str(e)}", exc_info=True)
        # Restituisci un errore generico al client, ma logga i dettagli
        raise HTTPException(status_code=500, detail=f"Errore interno del server durante l'elaborazione della query.")


@router.post("/reset", status_code=204)
async def reset_conversation_endpoint( # Rinomina
    engine: RAGEngine = Depends(get_rag_engine)
    # reset_request: Optional[ResetRequest] = None, # Per futura specificità di sessione
):
    """
    Resets the RAG engine's current conversation memory.
    """
    # session_id_to_reset = reset_request.session_id if reset_request else "global"
    logger.info(f"API call to reset conversation memory for the RAGEngine instance.")
    try:
        engine.reset_memory()
        # HTTP 204 No Content non richiede un corpo di risposta
    except Exception as e:
        logger.error(f"Error resetting memory via API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Errore durante il reset della memoria.")

@router.get("/transcript", response_model=TranscriptResponse)
async def get_conversation_transcript_endpoint( # Rinomina
    engine: RAGEngine = Depends(get_rag_engine)
    # session_id: Optional[str] = Query(None, description="ID della sessione per cui ottenere il transcript, se supportato.")
):
    """
    Retrieves the RAG engine's current conversation transcript.
    """
    logger.info(f"API call to retrieve transcript for the RAGEngine instance.")
    try:
        transcript_data = engine.get_transcript()
        return TranscriptResponse(transcript=transcript_data)
    except Exception as e:
        logger.error(f"Error retrieving transcript via API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Errore durante il recupero del transcript.")

@router.get("/contact", response_model=ContactInfo)
async def get_contact_info_endpoint(): # Rinomina
    """Get CRI contact information from application settings."""
    logger.info("API call to retrieve CRI contact information.")
    return ContactInfo(
        email=app_settings.CRI_CONTACT_EMAIL,
        phone=app_settings.CRI_CONTACT_PHONE,
        website=app_settings.CRI_WEBSITE,
    )