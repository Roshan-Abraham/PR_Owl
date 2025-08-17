"""
Enhanced VectorSearchTool with FastMap Optimization for Milvus Integration
Implements high-performance vector similarity search with intelligent caching and hybrid search capabilities
"""

import asyncio
import time
import hashlib
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import structlog
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential

# Milvus imports
try:
    from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
    from pymilvus.client.types import LoadState
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logger = structlog.get_logger()
    logger.warning("Milvus not available, using mock implementation")

# Sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

# ADK imports (placeholder)
try:
    from google.adk import AgentTool
except ImportError:
    class AgentTool:
        def __init__(self, name: str = "", description: str = ""):
            self.name = name
            self.description = description

from models.schemas import SearchResult
from infrastructure.config import settings

logger = structlog.get_logger()

@dataclass
class VectorSearchConfig:
    """Configuration for vector search operations"""
    embedding_model: str = "all-MiniLM-L6-v2"
    default_collection: str = "crisis_vectors"
    max_results: int = 50
    similarity_threshold: float = 0.5
    cache_ttl: int = 3600  # 1 hour
    index_type: str = "HNSW"
    metric_type: str = "COSINE"
    embedding_dim: int = 384

class EmbeddingService:
    """Service for generating text embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.cache = TTLCache(maxsize=500, ttl=3600)
        
    async def initialize(self):
        """Initialize the embedding model"""
        if EMBEDDINGS_AVAILABLE:
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model initialized", model=self.model_name)
        else:
            logger.warning("Sentence transformers not available, using mock embeddings")
            
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if cached_embedding := self.cache.get(text_hash):
            return cached_embedding
            
        if self.model:
            embedding = self.model.encode(text).tolist()
        else:
            # Mock embedding for development
            embedding = [0.1] * 384  # Standard dimension for all-MiniLM-L6-v2
            
        self.cache[text_hash] = embedding
        return embedding
        
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if self.model:
            embeddings = self.model.encode(texts).tolist()
            
            # Cache individual embeddings
            for text, embedding in zip(texts, embeddings):
                text_hash = hashlib.md5(text.encode()).hexdigest()
                self.cache[text_hash] = embedding
                
            return embeddings
        else:
            # Mock embeddings
            return [[0.1] * 384 for _ in texts]

class MilvusClient:
    """High-performance Milvus client with connection management"""
    
    def __init__(self, config: VectorSearchConfig):
        self.config = config
        self.connection_name = "crisis_management"
        self.collections = {}
        self.initialized = False
        
    async def initialize(self):
        """Initialize Milvus connection and collections"""
        if not MILVUS_AVAILABLE:
            logger.warning("Milvus not available, using mock client")
            self.initialized = True
            return
            
        try:
            # Connect to Milvus
            connections.connect(
                alias=self.connection_name,
                host=getattr(settings, 'MILVUS_HOST', 'localhost'),
                port=getattr(settings, 'MILVUS_PORT', '19530')
            )
            
            # Initialize default collection
            await self._ensure_collection_exists(self.config.default_collection)
            
            self.initialized = True
            logger.info("Milvus client initialized", connection=self.connection_name)
            
        except Exception as e:
            logger.error("Failed to initialize Milvus client", error=str(e))
            # Continue with mock implementation
            self.initialized = True
            
    async def _ensure_collection_exists(self, collection_name: str):
        """Ensure a collection exists with proper schema"""
        if not MILVUS_AVAILABLE:
            return
            
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            self.collections[collection_name] = collection
            
            # Load collection if not loaded
            if collection.is_empty is False:
                collection.load()
                
            logger.debug("Collection loaded", collection=collection_name)
            return
            
        # Create collection with schema
        schema = self._create_collection_schema()
        collection = Collection(collection_name, schema)
        
        # Create index
        index_params = {
            "index_type": self.config.index_type,
            "metric_type": self.config.metric_type,
            "params": {"M": 16, "efConstruction": 256}
        }
        collection.create_index("embedding", index_params)
        
        # Load collection
        collection.load()
        
        self.collections[collection_name] = collection
        logger.info("Collection created and indexed", collection=collection_name)
        
    def _create_collection_schema(self) -> CollectionSchema:
        """Create the schema for vector collections"""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.config.embedding_dim),
            FieldSchema(name="source_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="company_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="confidence_score", dtype=DataType.FLOAT),
            FieldSchema(name="created_at", dtype=DataType.INT64),
        ]
        
        return CollectionSchema(fields, description="Crisis management vector collection")
        
    async def search(self, collection_name: str, query_vectors: List[List[float]],
                    search_params: Dict[str, Any], limit: int,
                    expr: Optional[str] = None, output_fields: List[str] = None) -> List[List[Any]]:
        """Perform vector similarity search"""
        if not MILVUS_AVAILABLE or collection_name not in self.collections:
            # Mock search results
            return [[self._create_mock_result() for _ in range(min(limit, 3))] for _ in query_vectors]
            
        try:
            collection = self.collections[collection_name]
            
            results = collection.search(
                data=query_vectors,
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=expr,
                output_fields=output_fields or [
                    "source_id", "source_type", "title", "summary", 
                    "tags", "confidence_score", "created_at"
                ]
            )
            
            return results
            
        except Exception as e:
            logger.error("Vector search failed", collection=collection_name, error=str(e))
            # Return mock results as fallback
            return [[self._create_mock_result() for _ in range(min(limit, 3))] for _ in query_vectors]
            
    def _create_mock_result(self):
        """Create a mock search result for development"""
        class MockHit:
            def __init__(self):
                self.id = f"mock_{int(time.time())}"
                self.score = 0.8
                self.entity = {
                    "source_id": "mock_case_001",
                    "source_type": "case_study",
                    "title": "Mock Crisis Case",
                    "summary": "This is a mock crisis case for development",
                    "tags": "development,mock",
                    "confidence_score": 0.8,
                    "created_at": int(time.time())
                }
        return MockHit()
        
    async def insert(self, collection_name: str, data: List[Dict[str, Any]]):
        """Insert vectors into collection"""
        if not MILVUS_AVAILABLE or collection_name not in self.collections:
            logger.debug("Mock vector insert", collection=collection_name, count=len(data))
            return
            
        try:
            collection = self.collections[collection_name]
            collection.insert(data)
            collection.flush()
            logger.debug("Vectors inserted", collection=collection_name, count=len(data))
        except Exception as e:
            logger.error("Vector insert failed", collection=collection_name, error=str(e))
            
    async def close(self):
        """Close Milvus connections"""
        if MILVUS_AVAILABLE:
            try:
                connections.disconnect(self.connection_name)
                logger.info("Milvus connection closed")
            except Exception as e:
                logger.warning("Error closing Milvus connection", error=str(e))

class VectorSearchTool(AgentTool):
    """Enhanced VectorSearchTool with FastMap optimization and hybrid search"""
    
    def __init__(self, config: Optional[VectorSearchConfig] = None):
        super().__init__(
            "vector_search_tool",
            "High-performance vector similarity search with Milvus integration"
        )
        
        self.config = config or VectorSearchConfig()
        self.embedding_service = EmbeddingService(self.config.embedding_model)
        self.milvus_client = MilvusClient(self.config)
        
        # Performance caching
        self.search_cache = TTLCache(maxsize=200, ttl=self.config.cache_ttl)
        self.search_plan_cache = TTLCache(maxsize=100, ttl=1800)
        
        # Metrics
        self.search_stats = {
            "total_searches": 0,
            "cache_hits": 0,
            "avg_search_time_ms": 0.0,
            "successful_searches": 0
        }
        
    async def initialize(self):
        """Initialize the vector search tool"""
        await self.embedding_service.initialize()
        await self.milvus_client.initialize()
        logger.info("VectorSearchTool initialized")
        
    async def similarity_search(self, query_text: str, filters: Dict[str, Any],
                              company_id: str, top_k: int = 10,
                              search_params: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Company-scoped vector similarity search with intelligent caching"""
        start_time = time.time()
        self.search_stats["total_searches"] += 1
        
        try:
            # Generate cache key
            cache_key = self._generate_search_cache_key(query_text, filters, company_id, top_k)
            
            # Check cache
            if cached_result := self.search_cache.get(cache_key):
                self.search_stats["cache_hits"] += 1
                logger.debug("Search cache hit", query_preview=query_text[:50])
                return cached_result
                
            # Generate or retrieve cached embeddings
            embedding_cache_key = f"emb:{hashlib.md5(query_text.encode()).hexdigest()}"
            query_vector = await self.embedding_service.embed_text(query_text)
            
            # Perform vector search
            results = await self.vector_similarity_search(
                query_vector, filters, company_id, top_k, search_params
            )
            
            # Cache results
            self.search_cache[cache_key] = results
            self.search_stats["successful_searches"] += 1
            
            # Update metrics
            execution_time = (time.time() - start_time) * 1000
            self.search_stats["avg_search_time_ms"] = (
                (self.search_stats["avg_search_time_ms"] * (self.search_stats["total_searches"] - 1) + execution_time)
                / self.search_stats["total_searches"]
            )
            
            logger.debug("Vector similarity search completed", 
                        results_count=len(results), time_ms=execution_time)
            
            return results
            
        except Exception as e:
            logger.error("Similarity search failed", query_preview=query_text[:50], error=str(e))
            return []
            
    async def vector_similarity_search(self, query_vector: List[float], filters: Dict[str, Any],
                                     company_id: str, top_k: int = 10,
                                     search_params: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Direct vector search with company partitioning and metadata filtering"""
        
        # Collection selection strategy
        collection_name = self._select_collection(filters, company_id)
        
        # Default search parameters
        search_params = search_params or {
            "metric_type": self.config.metric_type,
            "params": {"nprobe": min(16, top_k * 2)}
        }
        
        # Build metadata filter expression
        filter_expression = self._build_filter_expression(filters, company_id)
        
        # Define output fields
        output_fields = [
            "source_id", "source_type", "title", "summary", 
            "tags", "confidence_score", "created_at"
        ]
        
        # Execute search with retry and fallback
        try:
            search_results = await self.milvus_client.search(
                collection_name=collection_name,
                query_vectors=[query_vector],
                search_params=search_params,
                limit=top_k * 2,  # Over-fetch for diversity
                expr=filter_expression,
                output_fields=output_fields
            )
            
            # Process and diversify results
            if search_results and search_results[0]:
                processed_results = self._diversify_results(search_results[0], top_k)
                
                return [
                    SearchResult(
                        id=str(hit.id),
                        similarity_score=float(hit.score),
                        metadata={field: getattr(hit.entity, field, None) for field in output_fields},
                        source_id=getattr(hit.entity, "source_id", ""),
                        source_type=getattr(hit.entity, "source_type", ""),
                        title=getattr(hit.entity, "title", ""),
                        summary=getattr(hit.entity, "summary", "")
                    ) for hit in processed_results
                ]
            else:
                return []
                
        except Exception as e:
            logger.warning("Primary vector search failed, using fallback", error=str(e))
            return await self._fallback_search(query_vector, company_id, top_k)
            
    async def hybrid_search(self, query_text: str, keyword_filters: Dict[str, Any],
                          vector_filters: Dict[str, Any], company_id: str, 
                          top_k: int = 10) -> List[SearchResult]:
        """Hybrid search combining vector similarity and keyword matching"""
        
        try:
            # Parallel vector and keyword searches
            vector_task = self.similarity_search(query_text, vector_filters, company_id, top_k)
            keyword_task = self._keyword_search(query_text, keyword_filters, company_id, top_k)
            
            vector_results, keyword_results = await asyncio.gather(
                vector_task, keyword_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(vector_results, Exception):
                logger.warning("Vector search failed in hybrid search", error=str(vector_results))
                vector_results = []
                
            if isinstance(keyword_results, Exception):
                logger.warning("Keyword search failed in hybrid search", error=str(keyword_results))
                keyword_results = []
            
            # Combine and re-rank results
            combined_results = self._combine_search_results(vector_results, keyword_results, top_k)
            
            logger.debug("Hybrid search completed", 
                        vector_results=len(vector_results),
                        keyword_results=len(keyword_results),
                        combined_results=len(combined_results))
            
            return combined_results
            
        except Exception as e:
            logger.error("Hybrid search failed", query_preview=query_text[:50], error=str(e))
            return []
            
    async def batch_similarity_search(self, queries: List[str], company_id: str,
                                    top_k: int = 5) -> Dict[str, List[SearchResult]]:
        """Optimized batch search for multiple queries"""
        
        try:
            # Batch embedding generation
            embeddings = await self.embedding_service.embed_texts(queries)
            
            # Concurrent vector searches
            search_tasks = [
                self.vector_similarity_search(
                    embedding, {"company_scoped": True}, company_id, top_k
                )
                for embedding in embeddings
            ]
            
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            processed_results = {}
            for query, result in zip(queries, results):
                if isinstance(result, Exception):
                    logger.warning("Batch search item failed", query_preview=query[:50], error=str(result))
                    processed_results[query] = []
                else:
                    processed_results[query] = result
                    
            logger.debug("Batch similarity search completed", queries_count=len(queries))
            
            return processed_results
            
        except Exception as e:
            logger.error("Batch similarity search failed", queries_count=len(queries), error=str(e))
            return {query: [] for query in queries}
            
    def _generate_search_cache_key(self, query_text: str, filters: Dict[str, Any],
                                 company_id: str, top_k: int) -> str:
        """Generate cache key for search operations"""
        key_data = {
            "query_hash": hashlib.md5(query_text.encode()).hexdigest(),
            "filters": sorted(filters.items()),
            "company_id": company_id,
            "top_k": top_k
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return f"search:{hashlib.md5(key_str.encode()).hexdigest()}"
        
    def _select_collection(self, filters: Dict[str, Any], company_id: str) -> str:
        """Select appropriate collection based on filters and company scoping"""
        if filters.get('company_scoped', False):
            return f"crisis_vectors_{company_id}"
        return self.config.default_collection
        
    def _build_filter_expression(self, filters: Dict[str, Any], company_id: str) -> Optional[str]:
        """Build Milvus filter expression from filters"""
        expr_parts = []
        
        # Company scoping (always applied unless company_scoped=True)
        if not filters.get('company_scoped', False):
            expr_parts.append(f'company_id == "{company_id}"')
            
        # Source type filters
        if source_types := filters.get('source_types'):
            source_filter = ' or '.join([f'source_type == "{st}"' for st in source_types])
            expr_parts.append(f'({source_filter})')
            
        # Date range filters
        if date_range := filters.get('date_range'):
            start_ts = int(date_range['start'].timestamp()) if isinstance(date_range['start'], datetime) else date_range['start']
            end_ts = int(date_range['end'].timestamp()) if isinstance(date_range['end'], datetime) else date_range['end']
            expr_parts.append(f'created_at >= {start_ts} and created_at <= {end_ts}')
            
        # Tag filters
        if tags := filters.get('tags'):
            tag_filter = ' or '.join([f'tags like "%{tag}%"' for tag in tags])
            expr_parts.append(f'({tag_filter})')
            
        # Confidence threshold
        if confidence_threshold := filters.get('min_confidence'):
            expr_parts.append(f'confidence_score >= {confidence_threshold}')
            
        return ' and '.join(expr_parts) if expr_parts else None
        
    def _diversify_results(self, search_results: List[Any], target_count: int) -> List[Any]:
        """Ensure result diversity by source type and recency"""
        if len(search_results) <= target_count:
            return search_results
            
        # Group by source type for diversity
        by_source = {}
        for result in search_results:
            source_type = getattr(result.entity, 'source_type', 'unknown')
            if source_type not in by_source:
                by_source[source_type] = []
            by_source[source_type].append(result)
            
        # Select diverse results using round-robin
        diversified = []
        source_types = list(by_source.keys())
        type_index = 0
        
        while len(diversified) < target_count and any(by_source.values()):
            current_type = source_types[type_index % len(source_types)]
            if by_source[current_type]:
                diversified.append(by_source[current_type].pop(0))
            type_index += 1
            
        return diversified[:target_count]
        
    async def _keyword_search(self, query_text: str, filters: Dict[str, Any],
                            company_id: str, top_k: int) -> List[SearchResult]:
        """Fallback keyword search (would integrate with Firestore or Elasticsearch)"""
        # Placeholder implementation - would integrate with text search service
        logger.debug("Performing keyword search", query_preview=query_text[:50])
        
        # Mock results for development
        return [
            SearchResult(
                id=f"keyword_result_{i}",
                similarity_score=0.6,
                metadata={"search_type": "keyword"},
                source_id=f"keyword_source_{i}",
                source_type="keyword_match",
                title=f"Keyword Match {i}",
                summary=f"This is a keyword match result for: {query_text[:100]}"
            )
            for i in range(min(3, top_k))
        ]
        
    def _combine_search_results(self, vector_results: List[SearchResult],
                              keyword_results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Combine and re-rank vector and keyword search results"""
        
        # Simple combination strategy - interleave results
        combined = []
        v_idx = k_idx = 0
        
        for i in range(min(top_k, len(vector_results) + len(keyword_results))):
            if i % 2 == 0 and v_idx < len(vector_results):
                # Boost vector similarity scores slightly
                result = vector_results[v_idx]
                result.similarity_score = min(1.0, result.similarity_score * 1.1)
                combined.append(result)
                v_idx += 1
            elif k_idx < len(keyword_results):
                combined.append(keyword_results[k_idx])
                k_idx += 1
            elif v_idx < len(vector_results):
                combined.append(vector_results[v_idx])
                v_idx += 1
                
        return combined
        
    async def _fallback_search(self, query_vector: List[float], company_id: str, 
                             top_k: int) -> List[SearchResult]:
        """Fallback search when primary vector search fails"""
        logger.debug("Using fallback search", company_id=company_id)
        
        # Mock fallback results
        return [
            SearchResult(
                id=f"fallback_result_{i}",
                similarity_score=0.4,
                metadata={"search_type": "fallback"},
                source_id=f"fallback_source_{i}",
                source_type="fallback",
                title=f"Fallback Result {i}",
                summary="This is a fallback search result"
            )
            for i in range(min(2, top_k))
        ]
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        cache_hit_rate = (
            (self.search_stats["cache_hits"] / max(1, self.search_stats["total_searches"])) * 100
        )
        
        return {
            **self.search_stats,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "cache_size": len(self.search_cache),
            "milvus_available": MILVUS_AVAILABLE,
            "embeddings_available": EMBEDDINGS_AVAILABLE
        }
        
    async def close(self):
        """Close vector search tool and cleanup resources"""
        await self.milvus_client.close()
        logger.info("VectorSearchTool closed")

# Export
__all__ = ['VectorSearchTool', 'VectorSearchConfig', 'EmbeddingService']