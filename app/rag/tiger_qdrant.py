"""
TigerQdrantStore - Una implementazione di VectorStore specializzata per la robustezza.
Sostituzione completa della classe QdrantVectorStore per una migliore gestione degli errori.
Progettata per il contesto specifico del "Tiger I" RAG Engine.
"""

from typing import Any, Dict, List, Optional, cast
import logging
import asyncio
from llama_index.core.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.schema import TextNode, Document
import qdrant_client
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest
from app.core.logging import get_logger

logger = get_logger(__name__)

class TigerQdrantStore(VectorStore):
    """Implementazione robusta di VectorStore per Qdrant."""
    
    stores_text: bool = True
    
    def __init__(
        self,
        client: qdrant_client.QdrantClient,
        aclient: Optional[AsyncQdrantClient] = None,
        collection_name: str = "crocerossa_docs",
        content_payload_key: str = "page_content",
        metadata_payload_key: str = "metadata",
    ) -> None:
        """Initialize with QdrantClient."""
        self._client = client
        self._aclient = aclient
        self._collection_name = collection_name
        self._content_payload_key = content_payload_key
        self._metadata_payload_key = metadata_payload_key
        logger.info(f"TigerQdrantStore initialized for collection '{collection_name}'")
        
    def _build_payload(self, node: TextNode) -> Dict[str, Any]:
        """Build payload from node."""
        metadata = node.metadata or {}
        return {
            self._content_payload_key: node.text,
            self._metadata_payload_key: metadata,
        }
        
    def _process_payload_robust(self, point_id: str, payload: Dict) -> tuple:
        """Process payload with robust NULL handling."""
        payload = payload or {}
        
        # Robust text content handling
        text_content = payload.get(self._content_payload_key, "")
        if text_content is None:
            logger.warning(f"TigerQdrantStore: Point {point_id} had NULL for '{self._content_payload_key}'. Converting to empty string.")
            text_content = ""
            
        # Robust metadata handling
        metadata = payload.get(self._metadata_payload_key, {})
        if metadata is None:
            logger.warning(f"TigerQdrantStore: Point {point_id} had NULL for metadata. Converting to empty dict.")
            metadata = {}
            
        return text_content, metadata
        
    def add(
        self,
        nodes: List[TextNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to Qdrant."""
        if not nodes:
            return []
        
        ids = []
        points = []
        
        for node in nodes:
            node_id = node.id_
            ids.append(node_id)
            
            point = rest.PointStruct(
                id=node_id,
                payload=self._build_payload(node),
                vector=node.embedding,
            )
            points.append(point)
            
        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
            **add_kwargs,
        )
        
        return ids
        
    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete using ref_doc_id."""
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=rest.FilterSelector(
                filter=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key=f"{self._metadata_payload_key}.ref_doc_id",
                            match=rest.MatchValue(value=ref_doc_id),
                        )
                    ]
                )
            ),
            **delete_kwargs,
        )
        
    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query Qdrant."""
        try:
            query_embedding = query.query_embedding
            if query_embedding is None:
                raise ValueError("Query embedding is None")
                
            response = self._client.search(
                collection_name=self._collection_name,
                query_vector=query_embedding,
                limit=query.similarity_top_k,
                with_payload=True,
                **kwargs,
            )
            
            nodes = []
            similarities = []
            ids = []
            
            for point in response:
                try:
                    point_id = str(point.id)
                    
                    # Process payload with robust NULL handling
                    text, metadata = self._process_payload_robust(point_id, point.payload)
                    
                    node = TextNode(
                        text=text,
                        metadata=metadata,
                        id_=point_id,
                    )
                    
                    nodes.append(node)
                    similarities.append(point.score)
                    ids.append(point_id)
                except Exception as e:
                    logger.error(f"Error processing point {point.id}: {e}")
                    # Skip this point and continue
                    
            return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
            
        except Exception as e:
            logger.error(f"Error in TigerQdrantStore.query: {e}", exc_info=True)
            # Return empty result on error
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
            
    async def aquery(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Async query to Qdrant."""
        if self._aclient is None:
            # Fallback to sync if async client not available
            return await asyncio.to_thread(self.query, query, **kwargs)
            
        try:
            query_embedding = query.query_embedding
            if query_embedding is None:
                raise ValueError("Query embedding is None")
                
            response = await self._aclient.search(
                collection_name=self._collection_name,
                query_vector=query_embedding,
                limit=query.similarity_top_k,
                with_payload=True,
                **kwargs,
            )
            
            nodes = []
            similarities = []
            ids = []
            
            for point in response:
                try:
                    point_id = str(point.id)
                    
                    # Process payload with robust NULL handling
                    text, metadata = self._process_payload_robust(point_id, point.payload)
                    
                    node = TextNode(
                        text=text,
                        metadata=metadata,
                        id_=point_id,
                    )
                    
                    nodes.append(node)
                    similarities.append(point.score)
                    ids.append(point_id)
                except Exception as e:
                    logger.error(f"Error processing point {point.id}: {e}")
                    # Skip this point and continue
                    
            return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
            
        except Exception as e:
            logger.error(f"Error in TigerQdrantStore.aquery: {e}", exc_info=True)
            # Return empty result on error
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])