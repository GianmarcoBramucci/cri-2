"""Test script to diagnose RAG engine issues"""
import os
from app.rag.engine import RAGEngine
from app.core.logging import configure_logging, get_logger

# Configure logging
configure_logging()
logger = get_logger(__name__)

# Aggiungi questo al test_rag.py
print("\nTesting Qdrant connection...")
from qdrant_client import QdrantClient
client = QdrantClient(url=os.environ.get("QDRANT_URL"), api_key=os.environ.get("QDRANT_API_KEY"))
collections = client.get_collections()
print(f"Available collections: {collections}")

print("Environment variables:")
# Print environment variables without showing actual values
for key in [
    "OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", 
    "QDRANT_COLLECTION", "COHERE_API_KEY"
]:
    print(f"  {key}: {'Present' if os.environ.get(key) else 'MISSING'}")

try:
    print("\nInitializing RAG Engine...")
    rag_engine = RAGEngine()
    print("RAG Engine initialized successfully!")
    
    # Test a simple query
    print("\nTesting query functionality...")
    result = rag_engine.query("Chi è la Croce Rossa?")
    print(f"Query response: {result['answer'][:100]}...")
    
    # Print the documents that were retrieved
    print("\nDocuments retrieved:")
    if 'documents' in result:
        for i, doc in enumerate(result['documents']):
            print(f"\nDocument {i+1}:")
            print(f"Content: {doc['content'][:200]}...")
            if 'metadata' in doc:
                print(f"Metadata: {doc['metadata']}")
            print(f"Score: {doc.get('score', 'N/A')}")
    elif 'source_documents' in result:
        for i, doc in enumerate(result['source_documents']):
            print(f"\nDocument {i+1}:")
            if hasattr(doc, 'page_content'):
                print(f"Content: {doc.page_content[:200]}...")
            elif 'content' in doc:
                print(f"Content: {doc['content'][:200]}...")
            else:
                print(f"Document structure: {type(doc)}")
            
            if hasattr(doc, 'metadata'):
                print(f"Metadata: {doc.metadata}")
            elif 'metadata' in doc:
                print(f"Metadata: {doc['metadata']}")
    else:
        print("No documents found in result. Available keys:", list(result.keys()))
    
except Exception as e:
    print(f"\nERROR: {str(e)}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc() 