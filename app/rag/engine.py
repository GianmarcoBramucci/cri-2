"""
RAG engine implementation for the CroceRossa Qdrant Cloud application.
Panzer VI Tiger I - Universal Production Ready Edition.
Clean async initialization and robust component handling.
"""

import asyncio
import json
import traceback
from typing import Dict, List, Optional, Any, Set
import re
import inspect

from llama_index.core import Settings as LlamaIndexSettings
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    QueryFusionRetriever,
    BaseRetriever
)

from app.core.logging import get_logger
logger = get_logger(__name__)

# Gestione più robusta dell'importazione di QueryFusionMode
FUSION_MODE_IMPORTED = False


from llama_index.core.schema import QueryBundle, NodeWithScore, TextNode
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.vector_stores.types import VectorStoreQueryResult 

from qdrant_client.http import models as rest 
from llama_index.embeddings.openai import OpenAIEmbedding 
from llama_index.llms.openai import OpenAI 
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core import VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
import qdrant_client
from qdrant_client import AsyncQdrantClient 
import qdrant_client.http.models as qmodels

from app.core.config import settings
from app.rag.memory import ConversationMemory
from app.rag.tiger_qdrant import TigerQdrantStore
from app.rag.prompts import (
    SYSTEM_PROMPT,
    CONDENSE_QUESTION_PROMPT,
    RAG_PROMPT,
    NO_CONTEXT_PROMPT,
)


class QdrantKeywordRetriever(BaseRetriever):
    def __init__(
        self,
        qdrant_client: qdrant_client.QdrantClient, 
        collection_name: str,
        content_payload_key: str = "page_content",  # Match your actual collection structure
        metadata_payload_key: str = "metadata",
        top_k: int = 10,
    ):
        self._client = qdrant_client
        self._collection_name = collection_name
        self._content_payload_key = content_payload_key
        self._metadata_payload_key = metadata_payload_key
        self._top_k = top_k
        super().__init__()
        
    def _get_keywords(self, query_str: str) -> List[str]:
        words = re.split(r'\W+', query_str.lower())
        keywords = [word for word in words if len(word) > 1 and word.isalpha()]
        seen = set()
        unique_keywords = [kw for kw in keywords if not (kw in seen or seen.add(kw))]
        logger.debug(f"Keywords for Qdrant FTS: {unique_keywords} from query: '{query_str}'")
        return unique_keywords
        
    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return await asyncio.to_thread(self._retrieve_sync, query_bundle)
        
    def _retrieve_sync(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        keywords = self._get_keywords(query_bundle.query_str)
        if not keywords:
            return []
            
        logger.info(f"Keyword FTS in '{self._collection_name}' on '{self._content_payload_key}' for: {keywords}")
        keyword_conditions = [
            qmodels.FieldCondition(key=self._content_payload_key, match=qmodels.MatchText(text=term))
            for term in keywords
        ]
        query_filter = qmodels.Filter(should=keyword_conditions)
        nodes_with_scores: List[NodeWithScore] = []
        
        try:
            scroll_fetch_limit = self._top_k * 3 
            offset = None
            retrieved_points_qdrant = []
            seen_ids = set()
            current_fetch_count = 0
            
            while current_fetch_count < scroll_fetch_limit:
                batch_limit = min(100, scroll_fetch_limit - current_fetch_count)
                if batch_limit <= 0:
                    break
                    
                response_points, next_offset = self._client.scroll(
                    collection_name=self._collection_name,
                    scroll_filter=query_filter,
                    limit=batch_limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                if not response_points:
                    break
                    
                for point in response_points:
                    if point.id not in seen_ids:
                        retrieved_points_qdrant.append(point)
                        seen_ids.add(point.id)
                        current_fetch_count += 1
                        if current_fetch_count >= scroll_fetch_limit:
                            break
                            
                if next_offset is None or current_fetch_count >= scroll_fetch_limit:
                    break
                    
                offset = next_offset
                
            logger.info(f"Keyword FTS initially retrieved {len(retrieved_points_qdrant)} points.")
            
            for point in retrieved_points_qdrant:
                payload = point.payload or {}
                text_content = payload.get(self._content_payload_key, "") 
                if text_content is None:
                    text_content = ""
                    logger.debug(f"KeywordRetriever: Point {point.id} had NULL for content key '{self._content_payload_key}'. Converting to empty string.")
                
                metadata_content = payload.get(self._metadata_payload_key, {})
                if metadata_content is None:
                    metadata_content = {}
                    logger.debug(f"KeywordRetriever: Point {point.id} had NULL for metadata key '{self._metadata_payload_key}'. Converting to empty dict.")
                
                num_matched_terms = sum(1 for term in keywords if term in text_content.lower())
                score = (float(num_matched_terms) / len(keywords)) if keywords and num_matched_terms > 0 else 0.01
                
                nodes_with_scores.append(NodeWithScore(
                    node=TextNode(
                        text=text_content,
                        metadata=metadata_content,
                        id_=str(point.id)
                    ),
                    score=score
                ))
                
            nodes_with_scores.sort(key=lambda x: x.score, reverse=True)
            return nodes_with_scores[:self._top_k]
            
        except Exception as e:
            logger.error(f"Error QdrantKeywordRetriever._retrieve_sync: {e}", exc_info=True)
            return []
            
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        logger.warning("QdrantKeywordRetriever._retrieve (sync) called.")
        return self._retrieve_sync(query_bundle)


class RAGEngine:
    def __init__(self, memory: Optional[ConversationMemory] = None):
        logger.info("RAGEngine instance created. Call ainitialize() for full component setup.")
        self._initialization_failed = True 
        self.memory = memory if memory is not None else ConversationMemory()
        self.qdrant_client: Optional[qdrant_client.QdrantClient] = None
        self.qdrant_aclient: Optional[AsyncQdrantClient] = None
        self.index: Optional[VectorStoreIndex] = None
        self.retriever: Optional[QueryFusionRetriever] = None
        self.dense_retriever: Optional[VectorIndexRetriever] = None  # Mantiene riferimento al dense retriever
        self.sparse_retriever: Optional[QdrantKeywordRetriever] = None  # Mantiene riferimento al sparse retriever
        self.reranker: Optional[CohereRerank] = None
        self.use_reranker: bool = False
        self._qdrant_initialized: bool = False
        self._fusion_retriever_enabled: bool = True  # Flag per indicare se il fusion retriever è abilitato
        # Update to match your actual Qdrant collection structure as shown in diagnostic
        self.content_payload_key_for_retrieval: str = "page_content" 
        self.metadata_payload_key_for_retrieval: str = "metadata"   
        self.condense_question_prompt = PromptTemplate(CONDENSE_QUESTION_PROMPT)
        self.qa_prompt = PromptTemplate(RAG_PROMPT)
        self.no_context_prompt = PromptTemplate(NO_CONTEXT_PROMPT)
        
        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "...":
            logger.critical("OPENAI_API_KEY not set. Cannot initialize LlamaIndex LLM/EmbedModel settings.")
            return 
            
        LlamaIndexSettings.llm = OpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.5,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=32768
        )
        LlamaIndexSettings.embed_model = OpenAIEmbedding(
            model_name=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY
        )
        logger.info("LlamaIndex global LLM and EmbedModel settings configured.")

    def _check_api_keys(self) -> List[str]:
        missing = []
        if not settings.QDRANT_API_KEY or settings.QDRANT_API_KEY == "...":
            missing.append("QDRANT_API_KEY")
        if not settings.QDRANT_URL or settings.QDRANT_URL == "...":
            missing.append("QDRANT_URL")
        if not settings.COHERE_API_KEY or settings.COHERE_API_KEY == "...":
            logger.info("COHERE_API_KEY not configured. Reranker disabled.")
        else:
            logger.info("COHERE_API_KEY configured. Reranker will be attempted.")
        return missing

    async def ainitialize(self) -> None:
        logger.info("RAGEngine - Tiger I - Async init components...")
        
        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "...":
            logger.critical("OPENAI_API_KEY not set. Aborting ainitialize.")
            self._initialization_failed = True
            return
            
        missing_qdrant_keys = self._check_api_keys()
        if any(key in ["QDRANT_API_KEY", "QDRANT_URL"] for key in missing_qdrant_keys):
            logger.critical(f"CRITICAL: Qdrant keys/URL missing. Setup aborted.")
            self._initialization_failed = True
            return
            
        try:
            logger.info(f"Connecting to Qdrant: URL='{settings.QDRANT_URL}'")
            
            self.qdrant_client = await asyncio.to_thread(
                qdrant_client.QdrantClient,
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=10.0
            )
            
            self.qdrant_aclient = AsyncQdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=10.0
            )
            
            collection_info = await asyncio.to_thread(
                self.qdrant_client.get_collection,
                settings.QDRANT_COLLECTION
            )
            
            logger.info(f"Qdrant connected. Collection '{settings.QDRANT_COLLECTION}' found.")
            logger.info(f"Using content_key='{self.content_payload_key_for_retrieval}', metadata_key='{self.metadata_payload_key_for_retrieval}'.")

            payload_schema = collection_info.payload_schema
            if self.content_payload_key_for_retrieval in payload_schema:
                field_info = payload_schema[self.content_payload_key_for_retrieval]
                if isinstance(field_info, qmodels.TextIndexParams):
                    logger.info(f"QDRANT CHECK: Field '{self.content_payload_key_for_retrieval}' FOUND and 'text' indexed.")
                else:
                    logger.warning(f"QDRANT CHECK: Field '{self.content_payload_key_for_retrieval}' FOUND, but NOT 'text' indexed (type: {type(field_info)}). Keyword search = SLOW.")
            else:
                logger.warning(f"QDRANT CHECK: Field '{self.content_payload_key_for_retrieval}' NOT in explicit schema. Assuming exists in payloads. Keyword search = SLOW.")

            # Utilizziamo la classe TigerQdrantStore per una gestione robusta dei NULL
            vector_store = TigerQdrantStore( 
                client=self.qdrant_client,
                aclient=self.qdrant_aclient,
                collection_name=settings.QDRANT_COLLECTION,
                content_payload_key=self.content_payload_key_for_retrieval,
                metadata_payload_key=self.metadata_payload_key_for_retrieval 
            )
            logger.info("Using TigerQdrantStore with specialized robust NULL handling for Tiger I engine reliability.")
            
            self.index = await asyncio.to_thread(VectorStoreIndex.from_vector_store, vector_store)
            
            # Salva i riferimenti ai retrievers individuali
            self.dense_retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=settings.RETRIEVAL_TOP_K
            )
            
            self.sparse_retriever = QdrantKeywordRetriever(
                qdrant_client=self.qdrant_client,
                collection_name=settings.QDRANT_COLLECTION,
                content_payload_key=self.content_payload_key_for_retrieval, 
                metadata_payload_key=self.metadata_payload_key_for_retrieval,
                top_k=settings.RETRIEVAL_TOP_K,
            )
            
            # SOLUZIONE SEMPLIFICATA: Inizializzazione QueryFusionRetriever senza parametro mode
            logger.info("Initializing QueryFusionRetriever without mode parameter")
            try:
                # Tenta di creare il retriever senza specificare il parametro mode
                self.retriever = QueryFusionRetriever(
                    retrievers=[self.dense_retriever, self.sparse_retriever],
                    similarity_top_k=settings.RETRIEVAL_TOP_K,
                    num_queries=1,
                )
                logger.info("Successfully created QueryFusionRetriever with default mode")
                self._fusion_retriever_enabled = True
            except Exception as e:
                logger.error(f"Error creating fusion retriever: {e}")
                # Se fallisce, usa solo il dense retriever come fallback
                logger.warning("Hybrid retriever failed to initialize. Using only dense retriever as fallback.")
                self.retriever = self.dense_retriever
                self._fusion_retriever_enabled = False
            
            if settings.COHERE_API_KEY and settings.COHERE_API_KEY != "...":
                try:
                    self.reranker = await asyncio.to_thread(
                        CohereRerank,
                        api_key=settings.COHERE_API_KEY,
                        top_n=settings.RERANK_TOP_K
                    )
                    self.use_reranker = True
                    logger.info(f"Cohere reranker enabled (top_n={settings.RERANK_TOP_K}).")
                except Exception as e_rerank:
                    logger.error(f"Cohere reranker init failed: {e_rerank}", exc_info=True)
                    self.use_reranker = False
            else:
                self.use_reranker = False
            
            self._qdrant_initialized = True
            self._initialization_failed = False 
            logger.info("RAGEngine async init complete!")
            
        except Exception as e:
            logger.error(f"FATAL: RAGEngine ainitialize error: {e}", exc_info=True)
            self._initialization_failed = True
            self._qdrant_initialized = False
            await self.aclose()

    async def _adirect_search_async(self, query: str) -> List[TextNode]:
        logger.warning(f"Async direct DENSE search: '{query}'")
        if not self._qdrant_initialized or not self.qdrant_aclient:
            logger.error("Async direct search: Qdrant aclient not initialized.")
            return []
            
        try:
            query_embedding = await LlamaIndexSettings.embed_model.aget_query_embedding(query)
            results = await self.qdrant_aclient.search(
                collection_name=settings.QDRANT_COLLECTION,
                query_vector=query_embedding,
                limit=settings.RETRIEVAL_TOP_K,
                with_payload=True
            )
            
            if not results:
                return []
            
            nodes = []
            for p in results:
                payload = p.payload or {}
                # Gestione robusta del text_content
                text_content = payload.get(self.content_payload_key_for_retrieval, "")
                if text_content is None:  # <-- Gestisce valori NULL
                    logger.debug(f"Direct search: Point {p.id} had NULL value for key '{self.content_payload_key_for_retrieval}'. Converting to empty string.")
                    text_content = ""
                    
                # Gestione robusta dei metadati
                metadata_content = payload.get(self.metadata_payload_key_for_retrieval, {})
                if metadata_content is None:  # <-- Gestisce metadati NULL
                    logger.debug(f"Direct search: Point {p.id} had NULL value for metadata. Converting to empty dict.")
                    metadata_content = {}
                
                # Crea il nodo solo se abbiamo un ID valido
                nodes.append(TextNode(
                    text=text_content,
                    metadata=metadata_content,
                    id_=str(p.id)
                ))
                    
            logger.info(f"Async direct search found {len(nodes)} valid results.")
            return nodes
            
        except Exception as e:
            logger.error(f"Error async direct search: {e}", exc_info=True)
            return []

    async def _acondense_question_async(self, question: str) -> str:
        if not self.memory.is_follow_up_question() or len(question.split()) <= 3:
            return question
            
        try:
            history = self.memory.get_history()
            if not history:
                return question
                
            chat_history_str = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in history])
            logger.info(f"Condensing. Model: {settings.CONDENSE_LLM_MODEL}. History: {len(history)}")
            
            condensation_llm = OpenAI(
                model=settings.CONDENSE_LLM_MODEL,
                api_key=settings.OPENAI_API_KEY,
                temperature=0.0
            )
            
            formatted_prompt = self.condense_question_prompt.format(
                chat_history=chat_history_str,
                question=question
            )
            
            messages = [ChatMessage(role=MessageRole.USER, content=formatted_prompt)]
            response = await condensation_llm.achat(messages) 
            condensed_text = response.message.content.strip()
            
            if not condensed_text or (len(condensed_text) < 5 and len(question) > 10):
                logger.warning(f"Condensed invalid vs original, using original.")
                return question
                
            if "?" not in condensed_text and "?" in question:
                condensed_text += "?"
                
            logger.info(f"Original: '{question}' -> Condensed: '{condensed_text}'")
            return condensed_text
            
        except Exception as e:
            logger.error(f"Error async condensing: {e}", exc_info=True)
            return question

    async def _aapply_reranking_async(self, query: str, nodes_with_scores: List[NodeWithScore]) -> List[TextNode]:
        if not self.use_reranker or not self.reranker or not nodes_with_scores or len(nodes_with_scores) <= 1:
            logger.info(f"Skipping rerank. Trimming to RERANK_TOP_K={settings.RERANK_TOP_K}.")
            return [nws.node for nws in nodes_with_scores[:settings.RERANK_TOP_K] if hasattr(nws, 'node')]
            
        try:
            logger.info(f"Async Cohere reranking {len(nodes_with_scores)} nodes.")
            reranked_nws = await asyncio.to_thread(
                self.reranker.postprocess_nodes,
                nodes_with_scores,
                query_str=query
            )
            
            final_nodes = [r_nws.node for r_nws in reranked_nws if hasattr(r_nws, 'node')]
            logger.info(f"Async reranking resulted in {len(final_nodes)} TextNodes.")
            return final_nodes
            
        except Exception as e:
            logger.error(f"Error async reranking: {e}", exc_info=True)
            logger.warning("Falling back to pre-rerank nodes (trimmed).")
            return [nws.node for nws in nodes_with_scores[:settings.RERANK_TOP_K] if hasattr(nws, 'node')]

    async def aquery(self, question: str, include_prompt: bool = False) -> Dict[str, Any]:
        logger.info(f"Async query: '{question}'")
        
        if self._initialization_failed or not self.retriever:
            error_msg = "RAG Engine not available/initialized."
            logger.error(error_msg)
            return {"answer": "Servizio RAG non disponibile.", "source_documents": [], "error": error_msg}
            
        condensed_q_text = await self._acondense_question_async(question)
        candidate_nodes: List[NodeWithScore] = []
        
        try:
            logger.info(f"Async hybrid retrieval for: '{condensed_q_text}'")
            if not self.retriever:
                raise RuntimeError("Retriever not initialized")
                
            # Tenta la ricerca tramite retriever ibrido
            try:
                if self._fusion_retriever_enabled:
                    candidate_nodes = await self.retriever.aretrieve(condensed_q_text)
                    # Filtra i nodi con testo None (che causerebbero errori)
                    candidate_nodes = [nws for nws in candidate_nodes if nws.node and nws.node.text is not None]
                    logger.info(f"Async hybrid retriever yielded {len(candidate_nodes)} candidates.")
                else:
                    # Se il fusion retriever è disabilitato, usa direttamente il dense
                    logger.info("Fusion retriever disabled. Using dense retriever directly.")
                    candidate_nodes = []
            except Exception as e_hybrid:
                # Se la ricerca ibrida fallisce, passa subito alla ricerca diretta
                logger.warning(f"Async hybrid retrieval failed: {e_hybrid}. Fallback.", exc_info=True)
                candidate_nodes = []
                
        except Exception as e_retrieve:
            logger.warning(f"Async hybrid retrieval failed: {e_retrieve}. Fallback.", exc_info=True)
            candidate_nodes = []
            
        final_ctx_nodes: List[TextNode] = []
        if candidate_nodes:
            final_ctx_nodes = await self._aapply_reranking_async(condensed_q_text, candidate_nodes)
        else: 
            logger.info("No candidates from hybrid. Async direct dense fallback.")
            fallback_nodes = await self._adirect_search_async(condensed_q_text)
            if fallback_nodes:
                fallback_nws = [NodeWithScore(node=tn, score=1.0/(i+1)) for i, tn in enumerate(fallback_nodes)]
                final_ctx_nodes = await self._aapply_reranking_async(condensed_q_text, fallback_nws)
                
        if not final_ctx_nodes:
            logger.warning(f"NO CONTEXT for: '{condensed_q_text}'")
            response_text = NO_CONTEXT_PROMPT 
            self.memory.add_exchange(question, response_text)
            return {"answer": response_text, "source_documents": [], "condensed_question": condensed_q_text}
            
        logger.info(f"Async LLM response with {len(final_ctx_nodes)} context nodes.")
        context_str = "\n\n".join([node.text for node in final_ctx_nodes if node.text is not None])
        chat_hist_str = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in self.memory.get_history()])
        
        final_prompt = self.qa_prompt.format(
            context=context_str,
            question=condensed_q_text,
            chat_history=chat_hist_str
        )
        
        if not LlamaIndexSettings.llm:
            logger.error("LLM not configured.")
            return {"answer": "Errore: LLM non configurato.", "source_documents": [], "error": "LLM_NOT_CONFIGURED"}
            
        llm_response = await LlamaIndexSettings.llm.acomplete(final_prompt)
        response_text = llm_response.text
        self.memory.add_exchange(question, response_text)
        
        sources_info = [
            {"text_preview": (n.text[:200]+"..."), "metadata": n.metadata or {}}
            for n in final_ctx_nodes if n.text is not None
        ]
        
        logger.info("Async query processed by Tiger I.")
        result = {
            "answer": response_text,
            "source_documents": sources_info,
            "condensed_question": condensed_q_text
        }
        
        if include_prompt:
            result["full_prompt"] = final_prompt
            
        return result

    def reset_memory(self) -> None:
        if hasattr(self, 'memory') and self.memory:
            self.memory.reset()
            logger.info(f"Memory reset (ID: {id(self.memory)}).")
        else:
            logger.warning("Reset memory: no memory object.")
            
    def get_transcript(self) -> List[Dict[str, str]]:
        if hasattr(self, 'memory') and self.memory:
            return self.memory.get_transcript()
        logger.warning("Get transcript: no memory object.")
        return []
        
    async def aclose(self):
        logger.info("Attempting RAGEngine resources cleanup...")
        
        if self.qdrant_aclient:
            try:
                await self.qdrant_aclient.close()
                logger.info("Async Qdrant client closed.")
            except Exception as e:
                logger.error(f"Error closing async Qdrant client: {e}")
                
        if self.qdrant_client and hasattr(self.qdrant_client, 'close') and callable(self.qdrant_client.close):
            try:
                self.qdrant_client.close()
                logger.info("Sync Qdrant client closed.")
            except Exception as e:
                logger.error(f"Error closing sync Qdrant client: {e}")
                
        logger.info("RAGEngine resources cleanup finished.")