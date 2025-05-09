"""Configuration management for the CroceRossa Qdrant Cloud application."""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    OPENAI_API_KEY: str = Field(..., description="OpenAI API key")
    QDRANT_URL: str = Field(..., description="Qdrant Cloud URL")
    QDRANT_API_KEY: str = Field(..., description="Qdrant Cloud API key")
    QDRANT_COLLECTION: str = Field(..., description="Qdrant Collection name")
    COHERE_API_KEY: str = Field(..., description="Cohere API key for reranking")
    
    # RAG Configuration
    RETRIEVAL_TOP_K: int = Field(70, description="Number of documents to retrieve")
    RERANK_TOP_K: int = Field(10, description="Number of documents to keep after reranking")
    MEMORY_WINDOW_SIZE: int = Field(4, description="Number of conversation exchanges to keep in memory")
    
    # LLM Configuration
    LLM_MODEL: str = Field("gpt-4.1", description="LLM model to use")
    EMBEDDING_MODEL: str = Field("text-embedding-3-large", description="Embedding model to use")
    
    # Miscellaneous
    LOG_LEVEL: str = Field("INFO", description="Logging level")
    ENVIRONMENT: str = Field("development", description="Application environment")
    
    # Contact Information
    CRI_CONTACT_EMAIL: str = Field("info@cri.it", description="CRI contact email")
    CRI_CONTACT_PHONE: str = Field("+39 06 47591", description="CRI contact phone")
    CRI_WEBSITE: str = Field("https://cri.it", description="CRI website")
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


# Create global settings instance
settings = Settings()