"""RAG engine implementation for the CroceRossa Qdrant Cloud application."""

import json
import traceback
from typing import Dict, List, Optional, Any

from llama_index.core import Settings as LlamaIndexSettings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core import VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
import qdrant_client
from llama_index.core.schema import TextNode

from app.core.config import settings
from app.core.logging import get_logger
from app.rag.memory import ConversationMemory
from app.rag.prompts import (
    SYSTEM_PROMPT,
    CONDENSE_QUESTION_PROMPT,
    RAG_PROMPT,
    NO_CONTEXT_PROMPT,
)
logger = get_logger(__name__)


class RAGEngine:
    """RAG Engine for the CroceRossa Qdrant Cloud application.
    
    This class handles:
    1. Connection to Qdrant Cloud
    2. Retrieval of relevant documents
    3. Reranking of results using Cohere
    4. Generation of responses using the LLM
    5. Conversation memory management
    """
    
    def __init__(self):
        """Initialize the RAG engine with the required components."""
        logger.info("Initializing RAG Engine")
        
        try:
            # Set the global LlamaIndex settings
            LlamaIndexSettings.llm = OpenAI(
                model=settings.LLM_MODEL,
                api_key=settings.OPENAI_API_KEY,
                temperature=0.1,
                system_prompt=SYSTEM_PROMPT,
            )
            
            LlamaIndexSettings.embed_model = OpenAIEmbedding(
                model_name=settings.EMBEDDING_MODEL,
                api_key=settings.OPENAI_API_KEY,
            )
            
            # Initialize conversation memory
            self.memory = ConversationMemory()
            
            # Connect to Qdrant
            self._initialize_qdrant()
            
            # Initialize prompt templates
            self.condense_question_prompt = PromptTemplate(CONDENSE_QUESTION_PROMPT)
            self.qa_prompt = PromptTemplate(RAG_PROMPT)
            self.no_context_prompt = PromptTemplate(NO_CONTEXT_PROMPT)
            
            logger.info("RAG Engine initialization complete")
        except Exception as e:
            logger.error("Failed to initialize RAG Engine", error=str(e))
            # Initialize memory to avoid NoneType errors in other methods
            self.memory = ConversationMemory()
            # Set flag to indicate initialization failure
            self._initialization_failed = True
    
    def _initialize_qdrant(self) -> None:
        """Initialize connection to Qdrant and set up the vector store."""
        try:
            logger.info("Connecting to Qdrant", 
                        url=settings.QDRANT_URL, 
                        collection=settings.QDRANT_COLLECTION)
            
            # Initialize Qdrant client
            self.qdrant_client = qdrant_client.QdrantClient(
                url=settings.QDRANT_URL, 
                api_key=settings.QDRANT_API_KEY
            )
            
            # Set up QdrantVectorStore with custom parameters
            # Imposta content_payload_key come None per evitare l'assunzione di un campo specifico
            vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=settings.QDRANT_COLLECTION,
                content_payload_key=None  # Ignora il campo del contenuto predefinito
            )
            
            # Create vector store index
            self.index = VectorStoreIndex.from_vector_store(vector_store)
            
            # Create retriever with top k
            self.retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=settings.RETRIEVAL_TOP_K,
            )
            
            # TEMPORARY: Skip using Cohere reranker due to compatibility issues
            self.use_reranker = False
            logger.info("Cohere reranker disabled for compatibility")
            
            logger.info("Qdrant and retrievers initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize Qdrant", error=str(e))
            raise
    
    def _direct_search(self, query: str) -> List[Dict]:
        """
        Esegue una ricerca diretta su Qdrant, bypassando LlamaIndex.
        
        Questa funzione è utile quando la struttura dei dati in Qdrant
        non è compatibile con le aspettative di LlamaIndex.
        """
        try:
            # Ottieni l'embedding per la query
            query_embedding = LlamaIndexSettings.embed_model.get_query_embedding(query)
            
            # Esegui la ricerca direttamente con il client Qdrant
            results = self.qdrant_client.search(
                collection_name=settings.QDRANT_COLLECTION,
                query_vector=query_embedding,
                limit=settings.RETRIEVAL_TOP_K,
                with_payload=True
            )
            
            if not results:
                logger.warning("No results found in direct search", query=query)
                return []
                
            logger.info(f"Found {len(results)} results in direct search")
            
            # Converti i risultati in nodi di testo
            nodes = []
            for point in results:
                # Estrai il payload
                if not hasattr(point, 'payload') or not point.payload:
                    continue
                    
                payload = point.payload
                
                # Determina quale campo usare come testo
                # Prova prima 'content', poi 'testo', poi 'documento'
                text_content = None
                for field in ['content', 'testo', 'documento', 'description', 'text']:
                    if field in payload and payload[field]:
                        text_content = payload[field]
                        break
                
                # Se non è stato trovato nessun campo di testo, unisci tutti i campi
                if text_content is None:
                    text_content = "\n".join([f"{k}: {v}" for k, v in payload.items() if v])
                
                # Crea il nodo di testo
                if text_content:
                    node = TextNode(
                        text=text_content,
                        metadata=payload
                    )
                    nodes.append(node)
            
            return nodes
            
        except Exception as e:
            logger.error("Error in direct search", error=str(e), stack_trace=traceback.format_exc())
            return []
    
    def _condense_question(self, question: str) -> str:
        """Condense a follow-up question using conversation history.
        
        Args:
            question: The current user question
            
        Returns:
            The condensed standalone question
        """
        # If there's no conversation history, return the original question
        if not self.memory.is_follow_up_question():
            return question
        
        try:
            # Format chat history for the prompt
            chat_history_str = ""
            for q, a in self.memory.get_history():
                chat_history_str += f"User: {q}\nAssistant: {a}\n\n"
            
            # Use LLM to condense the question
            llm = LlamaIndexSettings.llm
            messages = [
                ChatMessage(
                    role=MessageRole.USER,
                    content=self.condense_question_prompt.format(
                        chat_history=chat_history_str, 
                        question=question
                    )
                )
            ]
            
            response = llm.chat(messages)
            condensed_question = response.message.content
            
            logger.info("Condensed follow-up question", 
                      original=question, 
                      condensed=condensed_question)
            
            return condensed_question
            
        except Exception as e:
            logger.error("Failed to condense question", error=str(e))
            # Fall back to the original question if condensing fails
            return question
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process a user query and generate a response.
        
        Args:
            question: The user's question
            
        Returns:
            Dict containing the answer and metadata
        """
        logger.info("Processing query", question=question)
        
        try:
            # Check if initialization failed
            if hasattr(self, '_initialization_failed') and self._initialization_failed:
                error_message = "Mi dispiace, si è verificato un errore durante l'inizializzazione del sistema. Contatta il supporto tecnico."
                self.memory.add_exchange(question, error_message)
                return {
                    "answer": error_message,
                    "source_documents": [],
                    "error": "Initialization failed",
                }

            # Condense the question if it's a follow-up
            condensed_question = self._condense_question(question)
            
            # Usa la ricerca diretta invece di LlamaIndex retriever
            valid_nodes = self._direct_search(condensed_question)
            
            # Check if we have any valid results
            if not valid_nodes:
                logger.warning("No valid documents retrieved", question=condensed_question)
                # Use the no-context template
                response_text = LlamaIndexSettings.llm.complete(
                    self.no_context_prompt.format(question=condensed_question)
                ).text
                self.memory.add_exchange(question, response_text)
                return {
                    "answer": response_text,
                    "source_documents": [],
                    "condensed_question": condensed_question,
                }
            
            # Create context string from retrieved nodes
            context_str = "\n\n".join([
                f"Documento {i+1}:\n{node.text}" 
                for i, node in enumerate(valid_nodes)
            ])
            
            # Generate response
            prompt = self.qa_prompt.format(
                context=context_str,
                question=condensed_question
            )
            
            response_text = LlamaIndexSettings.llm.complete(prompt).text
            
            # Add to conversation memory
            self.memory.add_exchange(question, response_text)
            
            # Prepare source documents info
            source_docs = []
            for node in valid_nodes:
                if hasattr(node, 'metadata') and node.metadata:
                    source_docs.append({
                        "text": node.text[:200] + "...",  # Truncate for brevity
                        "metadata": node.metadata
                    })
            
            logger.info("Query processed successfully")
            
            return {
                "answer": response_text,
                "source_documents": source_docs,
                "condensed_question": condensed_question,
            }
            
        except Exception as e:
            logger.error("Error processing query", error=str(e), stack_trace=traceback.format_exc())
            error_message = "Mi dispiace, si è verificato un errore durante l'elaborazione della tua richiesta. Riprova più tardi o contatta il supporto tecnico."
            self.memory.add_exchange(question, error_message)
            return {
                "answer": error_message,
                "source_documents": [],
                "error": str(e),
            }
    
    def reset_memory(self) -> None:
        """Reset the conversation memory."""
        try:
            self.memory.reset()
            logger.info("Conversation memory reset")
        except Exception as e:
            logger.error("Error resetting memory", error=str(e))
    
    def get_transcript(self) -> List[Dict[str, str]]:
        """Get the conversation transcript.
        
        Returns:
            List of dictionaries with user questions and assistant answers
        """
        try:
            return self.memory.get_transcript()
        except Exception as e:
            logger.error("Error getting transcript", error=str(e))
            return []z