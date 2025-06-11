"""Main entry point for the CroceRossa Qdrant Cloud FastAPI application."""

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import time
import os
import asyncio 
from contextlib import asynccontextmanager 

from app.api.router import router as api_router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.rag.engine import RAGEngine 

configure_logging()
logger = get_logger("main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI application startup: Initializing RAG Engine via lifespan...")
    engine_instance = RAGEngine() 
    app.state.rag_engine = engine_instance 
    try:
        await engine_instance.ainitialize() 
        if engine_instance._initialization_failed:
            logger.critical("RAG Engine ainitialize() completed but reported an initialization failure during lifespan startup.")
        else:
            logger.info("RAG Engine instance successfully initialized via ainitialize() during lifespan startup and is ready.")
    except Exception as e:
        logger.critical(f"Fatal error during RAG Engine ainitialize() in lifespan startup: {e}", exc_info=True)
        if hasattr(app.state, 'rag_engine') and app.state.rag_engine:
             app.state.rag_engine._initialization_failed = True
    
    yield 
    
    logger.info("FastAPI application shutdown via lifespan...")
    if hasattr(app.state, 'rag_engine') and app.state.rag_engine:
        logger.info("Attempting to close RAG Engine resources...")
        await app.state.rag_engine.aclose()
    logger.info("Lifespan shutdown complete.")

app = FastAPI(
    title="CroceRossa Qdrant Cloud",
    description="Assistente virtuale conversazionale per la Croce Rossa Italiana - Edizione Tiger I Universal Production Ready",
    version="1.2.3", 
    lifespan=lifespan 
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

current_script_dir = os.path.dirname(os.path.abspath(__file__))
static_dir_path = os.path.join(current_script_dir, "static")

if not os.path.exists(static_dir_path): 
    logger.info(f"Static directory '{static_dir_path}' not found. Creating it.")
    os.makedirs(static_dir_path, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir_path), name="static")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled exception during request processing",
        error=str(exc), path=request.url.path, method=request.method, exc_info=True
    )
    return JSONResponse(status_code=500, content={"detail": "Si è verificato un errore interno del server."})

app.include_router(api_router, prefix="/api")

html_file_path_root = os.path.join(current_script_dir, "index.html")

@app.get("/")
async def serve_html_frontend():
    if not os.path.exists(html_file_path_root):
        logger.error(f"Frontend HTML file '{html_file_path_root}' not found at project root. Ensure it's in the same directory as main.py.")
        return JSONResponse(status_code=404, content={"detail": "Pagina principale non trovata."})
    with open(html_file_path_root, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), media_type="text/html")

@app.get("/health")
async def health_check():
    engine_status_message = "RAG Engine status not determined (instance not found in app state)."
    is_engine_healthy = False

    if hasattr(app.state, 'rag_engine') and app.state.rag_engine is not None:
        if app.state.rag_engine._initialization_failed:
            engine_status_message = "Error: RAG Engine initialization reported failure."
        else: 
            engine_status_message = "OK: RAG Engine initialized and operational."
            is_engine_healthy = True
    else:
        engine_status_message = "Critical Error: RAG Engine instance not created or found in application state."
        
    return {
        "application_status": "healthy" if is_engine_healthy else "unhealthy",
        "application_version": app.version, 
        "rag_engine_detail": engine_status_message
    }

if __name__ == "__main__":
    logger.info(f"Starting {app.title} v{app.version}", environment=settings.ENVIRONMENT)
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False, 
        log_level=settings.LOG_LEVEL.lower() 
    )