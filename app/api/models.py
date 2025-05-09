"""Pydantic models for the CroceRossa Qdrant Cloud API."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for the /query endpoint."""
    
    query: str = Field(..., description="The user's query about CRI")
    session_id: Optional[str] = Field(None, description="Optional session identifier")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default_factory=list, 
        description="Full conversation history from the client"
    )
    include_prompt: Optional[bool] = Field(
        False,
        description="Whether to include the full prompt in the response"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Come posso diventare volontario della Croce Rossa?",
                "session_id": "user_123456"
            }
        }


class QueryResponse(BaseModel):
    """Response model for the /query endpoint."""
    
    answer: str = Field(..., description="The assistant's response")
    source_documents: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="List of source documents used to generate the answer"
    )
    condensed_question: Optional[str] = Field(
        None, 
        description="The condensed version of the question if it was a follow-up"
    )
    full_prompt: Optional[str] = Field(
        None,
        description="The full prompt used to generate the answer"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Per diventare volontario della Croce Rossa Italiana devi...",
                "source_documents": [
                    {
                        "text": "Il percorso per diventare Volontario CRI prevede la frequenza...",
                        "metadata": {"source": "regolamento_volontari.pdf", "page": 12}
                    }
                ],
                "condensed_question": "Quali sono i requisiti e le procedure per diventare volontario della Croce Rossa Italiana?"
            }
        }


class ResetRequest(BaseModel):
    """Request model for the /reset endpoint."""
    
    session_id: Optional[str] = Field(None, description="Optional session identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "user_123456"
            }
        }


class ResetResponse(BaseModel):
    """Response model for the /reset endpoint."""
    
    success: bool = Field(..., description="Whether the reset was successful")
    message: str = Field(..., description="Status message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Conversazione resettata con successo."
            }
        }


class TranscriptResponse(BaseModel):
    """Response model for the /transcript endpoint."""
    
    transcript: List[Dict[str, str]] = Field(
        ..., 
        description="List of user questions and assistant answers"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "transcript": [
                    {
                        "user": "Come posso diventare volontario della Croce Rossa?",
                        "assistant": "Per diventare volontario della Croce Rossa Italiana devi..."
                    },
                    {
                        "user": "Quali corsi devo frequentare?",
                        "assistant": "È necessario frequentare il corso base..."
                    }
                ]
            }
        }


class ContactResponse(BaseModel):
    """Response model for the /contact endpoint."""
    
    name: str = Field(..., description="Organization name")
    website: str = Field(..., description="Organization website")
    email: str = Field(..., description="Contact email")
    phone: str = Field(..., description="Contact phone number")
    headquarters: str = Field(..., description="Headquarters address")
    description: str = Field(..., description="Brief description of the organization")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Croce Rossa Italiana",
                "website": "https://cri.it",
                "email": "info@cri.it",
                "phone": "+39 06 47591",
                "headquarters": "Via Bernardino Ramazzini, 31, 00151 Roma RM",
                "description": "La Croce Rossa Italiana, fondata il 15 giugno 1864, è un'associazione di soccorso volontario..."
            }
        }