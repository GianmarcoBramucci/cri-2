"""Main entry point for the CroceRossa Qdrant Cloud FastAPI application."""

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import time
import os

from app.api.router import router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger

# Configure logging
configure_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="CroceRossa Qdrant Cloud",
    description="Assistente virtuale conversazionale per la Croce Rossa Italiana",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    # Directory potrebbe non esistere ancora
    os.makedirs("static", exist_ok=True)
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Add request ID middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add request ID and process time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for the application."""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        path=request.url.path,
        method=request.method,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Si è verificato un errore interno. Riprova più tardi o contatta il supporto."
        },
    )

# Include API router
app.include_router(router, prefix="/api")

# Root endpoint che serve il file HTML
@app.get("/")
async def serve_html():
    """Serve the HTML frontend."""
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, media_type="text/html")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    logger.info(
        "Starting CroceRossa Qdrant Cloud application",
        environment=settings.ENVIRONMENT,
    )
    
    # Crea la directory static se non esiste
    os.makedirs("static", exist_ok=True)
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)