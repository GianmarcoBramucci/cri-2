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
    """RAG Engine for the CroceRossa Qdrant Cloud application."""
    
    def __init__(self, memory: ConversationMemory = None):
        """Initialize the RAG engine with the required components and an optional memory instance."""
        logger.info("Initializing RAG Engine" + (" with provided memory instance" if memory else ""))
        self._initialization_failed = False # Initialize the flag
        
        # Use the provided memory instance or create a new one
        self.memory = memory or ConversationMemory()
        
        self._qdrant_initialized = False  # Track Qdrant initialization
        
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
            
            # Connect to Qdrant
            self._initialize_qdrant()
            
            # Initialize prompt templates
            self.condense_question_prompt = PromptTemplate(CONDENSE_QUESTION_PROMPT)
            self.qa_prompt = PromptTemplate(RAG_PROMPT)
            self.no_context_prompt = PromptTemplate(NO_CONTEXT_PROMPT)
            
            logger.info("RAG Engine initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize RAG Engine: {str(e)}", exc_info=True)
            # Set flag to indicate initialization failure
            self._initialization_failed = True
    
    def _initialize_qdrant(self) -> None:
        """Initialize connection to Qdrant and set up the vector store with Cohere reranker."""
        # Skip if already initialized to prevent multiple connections
        if self._qdrant_initialized:
            logger.debug("Qdrant already initialized, skipping")
            return
            
        try:
            logger.info("Connecting to Qdrant", 
                        url=settings.QDRANT_URL, 
                        collection=settings.QDRANT_COLLECTION)
            
            # Initialize Qdrant client
            self.qdrant_client = qdrant_client.QdrantClient(
                url=settings.QDRANT_URL, 
                api_key=settings.QDRANT_API_KEY
            )
            
            # Set up QdrantVectorStore with correct content field
            vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=settings.QDRANT_COLLECTION,
                content_payload_key="page_content"  # <-- Qui è il fix principale!
            )
            
            # Create vector store index
            self.index = VectorStoreIndex.from_vector_store(vector_store)
            
            # Create retriever with top k
            self.retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=settings.RETRIEVAL_TOP_K,
            )
            
            # Initialize and enable Cohere reranker
            try:
                logger.info(f"Initializing Cohere reranker with top_k={settings.RERANK_TOP_K}")
                self.reranker = CohereRerank(
                    api_key=settings.COHERE_API_KEY,
                    top_n=settings.RERANK_TOP_K,
                    model_name="rerank-multilingual-v2.0"  # Supporta anche l'italiano
                )
                self.use_reranker = True
                logger.info("Cohere reranker initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Cohere reranker: {str(e)}", exc_info=True)
                self.use_reranker = False
                logger.warning("Cohere reranker disabled due to initialization failure")
            
            logger.info("Qdrant and retrievers initialized successfully")
            self._qdrant_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {str(e)}", exc_info=True)
            raise
    
    def _direct_search(self, query: str) -> List[TextNode]:
        """
        Esegue una ricerca diretta su Qdrant in caso di fallimento del retriever standard.
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
                logger.warning(f"No results found in direct search for query: {query}")
                return []
                
            logger.info(f"Found {len(results)} results in direct search")
            
            # Converti i risultati in nodi di testo
            nodes = []
            for point in results:
                if not hasattr(point, 'payload') or not point.payload:
                    continue
                
                payload = point.payload
                
                # Usa page_content come campo di testo
                if 'page_content' in payload and payload['page_content']:
                    node = TextNode(
                        text=payload['page_content'],
                        metadata=payload.get('metadata', {})
                    )
                    nodes.append(node)
            
            return nodes
            
        except Exception as e:
            logger.error(f"Error in direct search: {str(e)}", exc_info=True)
            return []
    
    def _validate_condensed_question(self, original: str, condensed: str) -> str:
        """Validate the condensed question to ensure it meets quality standards."""
        # Check if condensed question is empty or too short
        if not condensed or len(condensed) < 5:
            logger.warning(f"Condensed question too short or empty, using original: '{condensed}'")
            return original
            
        # Simple dictionary check for common Italian words to detect typos
        common_italian_words = ["elezioni", "maggio", "quando", "dove", "come", "giorno", "giornata", 
                                "elettorale", "votazioni", "voto", "urne", "seggio", "candidati", 
                                "croce", "rossa", "italiana", "nella", "sono", "stanno", "prossime"]
        
        # Check if specific words appear with typos
        words = condensed.split()
        for word in words:
            word_lower = word.lower().strip(",.?!;:")
            # Skip very short words
            if len(word_lower) <= 2:
                continue
                
            # Check if it's a common word with slight typos
            for common_word in common_italian_words:
                # If words are similar but not exact (likely typo)
                if word_lower != common_word and self._similar_words(word_lower, common_word, max_distance=2):
                    logger.warning(f"Possible typo detected: '{word_lower}' might be '{common_word}', using original question")
                    return original
                    
        # Check if condensed question contains obvious errors or typos (single letter words that aren't valid)
        suspicious_words = [w for w in words if len(w) == 1 and w.lower() not in ['a', 'e', 'è', 'o', 'i']]
        if suspicious_words:
            logger.warning(f"Suspicious single-letter words in condensed question: {suspicious_words}, using original")
            return original
            
        # Check for excessive typos (words with 4+ consonants in a row)
        consonants = "bcdfghjklmnpqrstvwxyz"
        for word in words:
            consonant_count = 0
            for char in word.lower():
                if char in consonants:
                    consonant_count += 1
                else:
                    consonant_count = 0
                if consonant_count >= 4:
                    logger.warning(f"Possible typo detected in condensed question: {word}, using original")
                    return original
                    
        # Count vowels - Italian words should have vowels
        vowels = "aeiouàèéìòù"
        all_vowels = sum(1 for char in condensed.lower() if char in vowels)
        if len(condensed) > 10 and all_vowels < len(condensed) * 0.2:  # Less than 20% vowels
            logger.warning(f"Question has too few vowels for Italian text: {condensed}, using original")
            return original
        
        # Everything looks good
        return condensed
        
    def _similar_words(self, word1: str, word2: str, max_distance: int = 2) -> bool:
        """Check if two words are similar using Levenshtein distance."""
        # Basic implementation of Levenshtein distance
        if abs(len(word1) - len(word2)) > max_distance:
            return False
            
        if len(word1) > 3 and len(word2) > 3:
            # For longer words, check if they start with the same characters
            prefix_len = min(3, min(len(word1), len(word2)))
            if word1[:prefix_len] != word2[:prefix_len]:
                return False
                
        # Calculate Levenshtein distance
        m, n = len(word1), len(word2)
        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
        
        for i in range(m+1):
            dp[i][0] = i
        for j in range(n+1):
            dp[0][j] = j
            
        for i in range(1, m+1):
            for j in range(1, n+1):
                cost = 0 if word1[i-1] == word2[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # deletion
                    dp[i][j-1] + 1,      # insertion
                    dp[i-1][j-1] + cost  # substitution
                )
        
        return dp[m][n] <= max_distance

    def _condense_question(self, question: str) -> str:
        """Condense a follow-up question using conversation history."""
        # Se non c'è storia o la domanda è molto breve, non riformulare
        if not self.memory.is_follow_up_question() or len(question.split()) <= 3:
            logger.info(f"Skipping condensation: no history or question too short: '{question}'")
            return question
        
        try:
            # Prepara la storia della conversazione per il prompt
            chat_history_str = ""
            # Utilizziamo TUTTA la storia recente, non solo le ultime 3 interazioni
            # per assicurarci che le informazioni personali vengano mantenute
            history = self.memory.get_history()
            
            if not history:
                logger.warning("No chat history available, using original question")
                return question
            
            # Elabora tutta la storia disponibile per catturare tutte le informazioni personali
            for q, a in history:
                chat_history_str += f"User: {q}\nAssistant: {a}\n\n"
            
            logger.info(f"Using {len(history)} exchanges for condensation to preserve personal details")
            
            # Imposta una temperatura più bassa per risposte più deterministiche
            condensation_llm = OpenAI(
                model=settings.LLM_MODEL,
                api_key=settings.OPENAI_API_KEY,
                temperature=0.0,
                system_prompt="Sei un assistente specializzato nella riformulazione di domande in italiano. Riformula la domanda di follow-up in una domanda autonoma, completa e chiara. Mantieni l'ortografia corretta. La domanda riformulata DEVE essere una frase completa e grammaticalmente corretta. ISTRUZIONE IMPORTANTE: Devi includere TUTTI i riferimenti a informazioni personali dell'utente (come nomi, preferenze, dettagli biografici) che sono stati menzionati in precedenza."
            )
            
            # Usa il prompt di condensazione
            prompt_content = self.condense_question_prompt.format(
                chat_history=chat_history_str, 
                question=question
            )
            
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content="Riformula la domanda in modo chiaro e completo. Includi sempre informazioni personali menzionate in precedenza."),
                ChatMessage(role=MessageRole.USER, content=prompt_content)
            ]
            
            response = condensation_llm.chat(messages)
            condensed_question = response.message.content.strip()
            
            # Validazione basilare
            if len(condensed_question) < 10 or "?" not in condensed_question:
                logger.warning("Condensed question seems invalid, using original")
                return question
            
            logger.info(f"Successfully condensed question: '{question}' → '{condensed_question}'")
            return condensed_question
            
        except Exception as e:
            logger.error(f"Error condensing question: {str(e)}")
            return question
    
    def _apply_reranking(self, query: str, nodes: List[TextNode]) -> List[TextNode]:
        """Applica il reranking ai nodi recuperati utilizzando Cohere."""
        if not self.use_reranker or not hasattr(self, 'reranker') or len(nodes) <= 1:
            logger.info("Skipping reranking: reranker disabled or not applicable")
            return nodes
            
        try:
            logger.info(f"Applying Cohere reranking to {len(nodes)} nodes")
            
            # Applica il reranker di Cohere
            reranked_nodes = self.reranker.postprocess(nodes, query_str=query)
            
            if reranked_nodes:
                logger.info(f"Successfully reranked nodes, keeping top {len(reranked_nodes)} of {len(nodes)}")
                return reranked_nodes
            else:
                logger.warning("Reranking returned empty results, using original nodes")
                return nodes
                
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}", exc_info=True)
            # In caso di errore, torna ai nodi originali
            return nodes
    
    def query(self, question: str, include_prompt: bool = False) -> Dict[str, Any]:
        """Process a user query and generate a response using instance-specific memory."""
        logger.info(f"Processing query with instance memory: '{question}'")
        
        try:
            # Check if initialization failed (flag set in __init__)
            if self._initialization_failed:
                error_message = "Mi dispiace, si è verificato un errore durante l'inizializzazione del sistema. Contatta il supporto tecnico."
                self.memory.add_exchange(question, error_message)
                return {
                    "answer": error_message,
                    "source_documents": [],
                    "error": "Initialization failed",
                }

            # Condense the question if it's a follow-up
            condensed_question = self._condense_question(question)
            
            # Tenta prima con il retriever standard
            try:
                retrieved_nodes = self.retriever.retrieve(condensed_question)
                valid_nodes = [node for node in retrieved_nodes if hasattr(node, 'text') and node.text]
            except Exception as e:
                logger.warning(f"Standard retriever failed, falling back to direct search: {str(e)}")
                valid_nodes = []
            
            # Se non abbiamo risultati validi, prova con la ricerca diretta
            if not valid_nodes:
                valid_nodes = self._direct_search(condensed_question)
                
            # Check if we have any valid results
            if not valid_nodes:
                logger.warning(f"No valid documents retrieved for question: {condensed_question}")
                # Use the no-context template
                prompt = self.no_context_prompt.format(
                    question=condensed_question,
                    chat_history="\n".join([f"User: {q}\nAssistant: {a}" for q, a in self.memory.get_history()])
                )
                response_text = LlamaIndexSettings.llm.complete(prompt).text
                self.memory.add_exchange(question, response_text)
                
                result = {
                    "answer": response_text,
                    "source_documents": [],
                    "condensed_question": condensed_question,
                }
                
                # Include il prompt completo se richiesto
                if include_prompt:
                    result["full_prompt"] = prompt
                
                return result
            
            # Applica il reranking ai nodi recuperati
            if self.use_reranker and len(valid_nodes) > 1:
                valid_nodes = self._apply_reranking(condensed_question, valid_nodes)
                logger.info(f"Using {len(valid_nodes)} nodes after reranking")
            
            # Create context string from retrieved nodes
            context_str = "\n\n".join([
                f"Documento {i+1}:\n{node.text}" 
                for i, node in enumerate(valid_nodes)
            ])
            
            # Generate response
            prompt = self.qa_prompt.format(
                context=context_str,
                question=condensed_question,
                chat_history="\n".join([f"User: {q}\nAssistant: {a}" for q, a in self.memory.get_history()])
            )
            
            response_text = LlamaIndexSettings.llm.complete(prompt).text
            
            # Add to conversation memory (self.memory is now session-specific)
            self.memory.add_exchange(question, response_text)
            
            # Prepare source documents info
            source_docs = []
            for node in valid_nodes:
                metadata = getattr(node, 'metadata', {}) or {}
                
                # Extract text preview safely
                node_text = getattr(node, 'text', '')
                text_preview = (node_text[:200] + "...") if node_text else "[Contenuto non disponibile]"
                
                source_docs.append({
                    "text": text_preview,
                    "metadata": metadata
                })
            
            logger.info("Query processed successfully")
            
            result = {
                "answer": response_text,
                "source_documents": source_docs,
                "condensed_question": condensed_question,
            }
            
            # Include il prompt completo se richiesto
            if include_prompt:
                result["full_prompt"] = prompt
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            error_message = "Mi dispiace, si è verificato un errore durante l'elaborazione della tua richiesta. Riprova più tardi o contatta il supporto tecnico."
            self.memory.add_exchange(question, error_message)
            return {
                "answer": error_message,
                "source_documents": [],
                "error": str(e),
            }
    
    def reset_memory(self) -> None:
        """Reset the conversation memory for the current session."""
        try:
            if self.memory: # Check if memory object exists
                self.memory.reset()
                logger.info(f"Conversation memory reset for the current session (memory ID: {id(self.memory)})")
            else:
                logger.warning("Attempted to reset memory, but no memory object was found on RAGEngine instance.")
        except Exception as e:
            logger.error(f"Error resetting memory: {str(e)}", exc_info=True)
    
    def get_transcript(self) -> List[Dict[str, str]]:
        """Get the conversation transcript for the current session."""
        try:
            if self.memory:
                transcript = self.memory.get_transcript()
                logger.info(f"Retrieved transcript with {len(transcript)} exchanges")
                return transcript
            logger.warning("Attempted to get transcript, but no memory object was found.")
            return []
        except Exception as e:
            logger.error(f"Error getting transcript: {str(e)}", exc_info=True)
            return []