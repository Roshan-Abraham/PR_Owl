"""
FastMap-Optimized Firestore Client with Connection Pooling and Intelligent Caching
Implements high-performance Firestore operations for the Crisis Management System
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import json
import structlog
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import AsyncClient, AsyncCollectionReference, AsyncDocumentReference
from google.cloud.firestore_v1.base_query import FieldFilter
from google.api_core import exceptions as gcp_exceptions

from models.schemas import QueryFilter, QueryOptions

logger = structlog.get_logger()


class SecurityError(Exception):
    """Raised when company access validation fails"""
    pass


class ValidationError(Exception):
    """Raised when document validation fails"""
    pass


class ConnectionPool:
    """Async connection pool for Firestore clients"""

    def __init__(self, min_connections: int = 5, max_connections: int = 20):
        self.min_connections = min_connections
        self.max_connections = max_connections
        self._pool: List[AsyncClient] = []
        self._in_use: set = set()
        self._lock = asyncio.Lock()
        self._created_count = 0

    async def initialize(self):
        """Initialize the connection pool with minimum connections"""
        for _ in range(self.min_connections):
            client = await self._create_client()
            self._pool.append(client)

    async def _create_client(self) -> AsyncClient:
        """Create a new Firestore client"""
        try:
            # Initialize Firebase Admin if not already done
            if not firebase_admin._apps:
                # Use default credentials for development
                # In production, use service account key
                cred = credentials.ApplicationDefault()
                firebase_admin.initialize_app(cred)

            client = firestore.AsyncClient()
            self._created_count += 1
            logger.debug("Created new Firestore client",
                         total_clients=self._created_count)
            return client

        except Exception as e:
            logger.error("Failed to create Firestore client", error=str(e))
            raise

    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool"""
        client = None
        async with self._lock:
            if self._pool:
                client = self._pool.pop()
                self._in_use.add(client)
            elif len(self._in_use) < self.max_connections:
                client = await self._create_client()
                self._in_use.add(client)

        if client is None:
            # Pool exhausted, wait for a connection
            await asyncio.sleep(0.1)
            async with self.get_connection() as conn:
                yield conn
            return

        try:
            yield client
        finally:
            async with self._lock:
                if client in self._in_use:
                    self._in_use.remove(client)
                if len(self._pool) < self.min_connections:
                    self._pool.append(client)

    async def close(self):
        """Close all connections in the pool"""
        async with self._lock:
            for client in self._pool + list(self._in_use):
                try:
                    await client.close()
                except Exception as e:
                    logger.warning(
                        "Error closing Firestore client", error=str(e))
            self._pool.clear()
            self._in_use.clear()

    @property
    def active_connections(self) -> int:
        """Get the number of active connections"""
        return len(self._in_use)


class QueryPlan:
    """Represents an optimized query execution plan"""

    def __init__(self, filters: List[QueryFilter], options: Optional[QueryOptions] = None):
        self.filters = filters
        self.options = options or QueryOptions()
        self.filter_groups = self._optimize_filters()

    def _optimize_filters(self) -> List[List[QueryFilter]]:
        """Optimize filter ordering for compound indexes"""
        # Group filters by type for optimal index usage
        equality_filters = []
        range_filters = []
        array_filters = []

        for f in self.filters:
            if f.operator == "==":
                equality_filters.append(f)
            elif f.operator in [">=", "<=", ">", "<"]:
                range_filters.append(f)
            elif f.operator in ["array-contains", "in"]:
                array_filters.append(f)

        # Firestore requires equality filters first, then range filters
        return [equality_filters, range_filters, array_filters]


class FirestoreConnectionPool:
    """
    High-level Firestore client with connection pooling, caching, and performance optimization
    """

    def __init__(self, min_connections: int = 5, max_connections: int = 20):
        self.pool = ConnectionPool(min_connections, max_connections)

        # Multi-level caching strategy
        self.document_cache = TTLCache(
            maxsize=1000, ttl=600)  # 10-minute default
        self.query_cache = TTLCache(
            maxsize=500, ttl=300)      # 5-minute query cache
        self.query_plan_cache = TTLCache(
            maxsize=200, ttl=900)  # 15-minute plan cache

        # Performance metrics
        self.query_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_queries": 0,
            "avg_query_time_ms": 0.0
        }

    async def initialize(self):
        """Initialize the connection pool"""
        await self.pool.initialize()
        logger.info("Firestore connection pool initialized")

    async def close(self):
        """Close the connection pool"""
        await self.pool.close()
        logger.info("Firestore connection pool closed")

    def _generate_cache_key(self, collection_path: str, doc_id: str = None,
                            filters: List[QueryFilter] = None,
                            options: QueryOptions = None) -> str:
        """Generate a cache key for the operation"""
        if doc_id:
            return f"doc:{collection_path}:{doc_id}"
        else:
            filter_str = json.dumps([(f.field, f.operator, f.value)
                                    for f in filters or []], sort_keys=True)
            options_str = json.dumps(
                options.model_dump() if options else {}, sort_keys=True)
            return f"query:{collection_path}:{hash(filter_str + options_str)}"

    def _calculate_adaptive_ttl(self, collection_path: str) -> int:
        """Calculate TTL based on data volatility patterns"""
        if 'dashboard' in collection_path:
            return 60  # High volatility - 1 minute
        elif 'crises' in collection_path:
            return 300  # Medium volatility - 5 minutes
        elif 'companies' in collection_path:
            return 1800  # Low volatility - 30 minutes
        return 600  # Default - 10 minutes

    def _validate_company_access(self, collection_path: str, company_id: str, doc_data: Dict = None) -> bool:
        """Validate that the operation is scoped to the correct company"""
        # Allow operations on company's own collections
        if collection_path.startswith(f"companies/{company_id}"):
            return True

        # For other collections, check if document contains company_id
        if doc_data and doc_data.get("company_id") == company_id:
            return True

        # Global collections that don't require company scoping
        global_collections = ["agent_runs", "vector_metadata"]
        if any(collection_path.startswith(gc) for gc in global_collections):
            return True

        return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_document(self, document_path: str, company_id: str = None) -> Optional[Dict[str, Any]]:
        """Get a single document with caching and company validation"""
        start_time = time.time()

        try:
            # Check cache first
            cache_key = self._generate_cache_key(document_path)
            if cached_result := self.document_cache.get(cache_key):
                self.query_stats["cache_hits"] += 1
                logger.debug("Cache hit for document", path=document_path)
                return cached_result

            self.query_stats["cache_misses"] += 1

            async with self.pool.get_connection() as client:
                doc_ref = client.document(document_path)
                doc_snapshot = await doc_ref.get()

                if not doc_snapshot.exists:
                    return None

                data = doc_snapshot.to_dict()
                data['_firestore_id'] = doc_snapshot.id
                data['_firestore_path'] = document_path
                data['_last_updated'] = doc_snapshot.update_time

                # Company access validation
                if company_id and not self._validate_company_access(document_path, company_id, data):
                    raise SecurityError(
                        f"Unauthorized access to {document_path} for company {company_id}")

                # Cache with adaptive TTL
                ttl = self._calculate_adaptive_ttl(document_path)
                self.document_cache[cache_key] = data

                execution_time = (time.time() - start_time) * 1000
                logger.debug("Document retrieved",
                             path=document_path, time_ms=execution_time)

                return data

        except Exception as e:
            logger.error("Failed to get document",
                         path=document_path, error=str(e))
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def set_document(self, document_path: str, data: Dict[str, Any],
                           company_id: str = None, merge: bool = False) -> None:
        """Set a document with company validation"""
        try:
            # Add system metadata
            now = datetime.utcnow()
            if not merge:
                data['created_at'] = now
            data['updated_at'] = now

            # Company access validation
            if company_id and not self._validate_company_access(document_path, company_id, data):
                raise SecurityError(
                    f"Unauthorized write to {document_path} for company {company_id}")

            async with self.pool.get_connection() as client:
                doc_ref = client.document(document_path)

                if merge:
                    await doc_ref.set(data, merge=True)
                else:
                    await doc_ref.set(data)

                # Invalidate cache
                cache_key = self._generate_cache_key(document_path)
                self.document_cache.pop(cache_key, None)

                logger.debug("Document set", path=document_path)

        except Exception as e:
            logger.error("Failed to set document",
                         path=document_path, error=str(e))
            raise

    async def create_document(self, document_path: str, data: Dict[str, Any],
                              company_id: str = None) -> None:
        """Create a new document"""
        await self.set_document(document_path, data, company_id, merge=False)

    async def update_document(self, document_path: str, updates: Dict[str, Any],
                              company_id: str = None) -> None:
        """Update specific fields in a document"""
        await self.set_document(document_path, updates, company_id, merge=True)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def query_collection(self, collection_path: str, filters: List[Tuple[str, str, Any]] = None,
                               order_by: Tuple[str, str] = None, limit: int = None,
                               company_id: str = None) -> List[Dict[str, Any]]:
        """Query a collection with optimization and caching"""
        start_time = time.time()

        try:
            # Convert tuple filters to QueryFilter objects
            query_filters = []
            if filters:
                for field, operator, value in filters:
                    query_filters.append(QueryFilter(
                        field=field, operator=operator, value=value))

            options = QueryOptions()
            if order_by:
                options.order_by = order_by[0]
                options.order_direction = order_by[1]
            if limit:
                options.limit = limit

            # Check cache
            cache_key = self._generate_cache_key(
                collection_path, filters=query_filters, options=options)
            if cached_result := self.query_cache.get(cache_key):
                self.query_stats["cache_hits"] += 1
                logger.debug("Query cache hit", collection=collection_path)
                return cached_result

            self.query_stats["cache_misses"] += 1
            self.query_stats["total_queries"] += 1

            async with self.pool.get_connection() as client:
                query = client.collection(collection_path)

                # Apply company scoping if required
                if company_id and not collection_path.startswith(f"companies/{company_id}"):
                    query = query.where("company_id", "==", company_id)

                # Apply filters
                if query_filters:
                    plan = QueryPlan(query_filters, options)
                    for filter_group in plan.filter_groups:
                        for f in filter_group:
                            query = query.where(f.field, f.operator, f.value)

                # Apply ordering
                if options.order_by:
                    direction = firestore.Query.ASCENDING if options.order_direction == "asc" else firestore.Query.DESCENDING
                    query = query.order_by(
                        options.order_by, direction=direction)

                # Apply limit
                if options.limit:
                    query = query.limit(options.limit)

                # Execute query
                results = []
                async for doc in query.stream():
                    data = doc.to_dict()
                    data['_firestore_id'] = doc.id
                    data['_firestore_path'] = f"{collection_path}/{doc.id}"
                    results.append(data)

                # Cache results
                cache_ttl = self._calculate_adaptive_ttl(collection_path)
                self.query_cache.set(cache_key, results, ttl=cache_ttl)

                execution_time = (time.time() - start_time) * 1000
                self.query_stats["avg_query_time_ms"] = (
                    (self.query_stats["avg_query_time_ms"] *
                     (self.query_stats["total_queries"] - 1) + execution_time)
                    / self.query_stats["total_queries"]
                )

                logger.debug("Collection queried", collection=collection_path,
                             results_count=len(results), time_ms=execution_time)

                return results

        except Exception as e:
            logger.error("Failed to query collection",
                         collection=collection_path, error=str(e))
            raise

    async def batch_write(self, operations: List[Dict[str, Any]], company_id: str = None) -> None:
        """Execute multiple write operations in a batch"""
        try:
            async with self.pool.get_connection() as client:
                batch = client.batch()

                for op in operations:
                    op_type = op.get("type")
                    path = op.get("path")
                    data = op.get("data", {})

                    # Company validation
                    if company_id and not self._validate_company_access(path, company_id, data):
                        raise SecurityError(
                            f"Unauthorized batch operation on {path} for company {company_id}")

                    doc_ref = client.document(path)

                    if op_type == "set":
                        batch.set(doc_ref, data)
                    elif op_type == "update":
                        batch.update(doc_ref, data)
                    elif op_type == "delete":
                        batch.delete(doc_ref)

                await batch.commit()

                # Invalidate relevant caches
                for op in operations:
                    cache_key = self._generate_cache_key(op["path"])
                    self.document_cache.pop(cache_key, None)

                logger.debug("Batch write completed",
                             operations_count=len(operations))

        except Exception as e:
            logger.error("Batch write failed", error=str(e))
            raise

    async def transaction_write(self, transaction_func, company_id: str = None, max_retries: int = 3) -> Any:
        """Execute a transaction with retry logic"""
        for attempt in range(max_retries):
            try:
                async with self.pool.get_connection() as client:
                    transaction = client.transaction()
                    return await transaction_func(transaction, client)

            except gcp_exceptions.Aborted as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                logger.warning("Transaction aborted, retrying",
                               attempt=attempt + 1)

            except Exception as e:
                logger.error("Transaction failed",
                             attempt=attempt + 1, error=str(e))
                raise

    async def health_check(self) -> bool:
        """Check database connectivity"""
        try:
            async with self.pool.get_connection() as client:
                # Simple connectivity test
                test_doc = client.document("_health_check/test")
                await test_doc.get()
                return True
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return False

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        cache_hit_rate = (
            self.query_stats["cache_hits"] /
            max(1, self.query_stats["cache_hits"] +
                self.query_stats["cache_misses"])
        ) * 100

        return {
            **self.query_stats,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "active_connections": self.pool.active_connections,
            "cache_sizes": {
                "document_cache": len(self.document_cache),
                "query_cache": len(self.query_cache),
                "query_plan_cache": len(self.query_plan_cache)
            }
        }

    def clear_caches(self):
        """Clear all caches (useful for testing)"""
        self.document_cache.clear()
        self.query_cache.clear()
        self.query_plan_cache.clear()
        logger.info("All caches cleared")
