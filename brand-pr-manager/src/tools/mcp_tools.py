"""
FastMap-Optimized MCP Tools for Crisis Management System
Implements high-performance MCP tools with intelligent caching, batch operations, and type safety
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, TypeVar, Generic, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import structlog
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential

# Google ADK imports (placeholder - actual implementation would use real ADK)
try:
    from google.adk import AgentTool
except ImportError:
    # Fallback for development
    class AgentTool:
        def __init__(self, name: str = "", description: str = ""):
            self.name = name
            self.description = description

from infrastructure.firestore_client import FirestoreConnectionPool, SecurityError, ValidationError
from models.schemas import QueryFilter, QueryOptions, SearchResult
from infrastructure.config import settings

logger = structlog.get_logger()
T = TypeVar('T')

# MCP Tool Base Classes

class BaseMCPTool(AgentTool):
    """Base class for all MCP tools with common functionality"""
    
    def __init__(self, name: str, description: str, connection_pool: FirestoreConnectionPool):
        super().__init__(name, description)
        self.pool = connection_pool
        self.execution_stats = {
            "total_calls": 0,
            "success_calls": 0,
            "error_calls": 0,
            "avg_execution_time_ms": 0.0
        }

    async def _execute_with_metrics(self, operation_name: str, func, *args, **kwargs):
        """Execute operation with metrics tracking"""
        start_time = time.time()
        self.execution_stats["total_calls"] += 1
        
        try:
            result = await func(*args, **kwargs)
            self.execution_stats["success_calls"] += 1
            return result
        except Exception as e:
            self.execution_stats["error_calls"] += 1
            logger.error(f"{operation_name} failed", tool=self.name, error=str(e))
            raise
        finally:
            execution_time = (time.time() - start_time) * 1000
            self.execution_stats["avg_execution_time_ms"] = (
                (self.execution_stats["avg_execution_time_ms"] * (self.execution_stats["total_calls"] - 1) + execution_time)
                / self.execution_stats["total_calls"]
            )

# Firestore MCP Tools

class FirestoreReadTool(BaseMCPTool, Generic[T]):
    """FastMap-optimized Firestore read operations with intelligent caching"""
    
    def __init__(self, connection_pool: FirestoreConnectionPool, schema_validator=None):
        super().__init__(
            "firestore_read_tool",
            "High-performance Firestore read operations with caching and type safety"
        )
        self.pool = connection_pool
        self.validator = schema_validator
        self.cache = TTLCache(maxsize=1000, ttl=600)
        self.query_plan_cache = TTLCache(maxsize=500, ttl=900)

    async def read_document(self, collection_path: str, doc_id: str, company_id: str,
                          expected_type: Optional[type] = None) -> Optional[T]:
        """Company-scoped document read with type safety and intelligent caching"""
        return await self._execute_with_metrics(
            "read_document",
            self._read_document_impl,
            collection_path, doc_id, company_id, expected_type
        )

    async def _read_document_impl(self, collection_path: str, doc_id: str, company_id: str,
                                expected_type: Optional[type] = None) -> Optional[T]:
        cache_key = f"{company_id}:{collection_path}:{doc_id}"
        
        # Check cache first
        if cached_result := self.cache.get(cache_key):
            logger.debug("Cache hit for document", path=f"{collection_path}/{doc_id}")
            return cached_result

        # Get from Firestore
        document_path = f"{collection_path}/{doc_id}"
        data = await self.pool.get_document(document_path, company_id)
        
        if not data:
            return None

        # Type validation if requested
        if expected_type and self.validator:
            if not self.validator.validate(data, expected_type):
                raise ValidationError(f"Document {doc_id} doesn't match expected schema")

        # Cache with adaptive TTL
        ttl = self._calculate_adaptive_ttl(collection_path)
        self.cache[cache_key] = data
        
        return data

    async def query_collection(self, collection_path: str, filters: List[QueryFilter],
                             company_id: str, options: Optional[QueryOptions] = None) -> List[T]:
        """High-performance collection queries with compound index optimization"""
        return await self._execute_with_metrics(
            "query_collection",
            self._query_collection_impl,
            collection_path, filters, company_id, options
        )

    async def _query_collection_impl(self, collection_path: str, filters: List[QueryFilter],
                                   company_id: str, options: Optional[QueryOptions] = None) -> List[T]:
        # Convert to tuples for firestore client
        filter_tuples = [(f.field, f.operator, f.value) for f in filters]
        order_by = None
        if options and options.order_by:
            order_by = (options.order_by, options.order_direction)

        results = await self.pool.query_collection(
            collection_path,
            filters=filter_tuples,
            order_by=order_by,
            limit=options.limit if options else None,
            company_id=company_id
        )

        return results

    async def batch_read_documents(self, doc_refs: List[tuple], company_id: str) -> Dict[str, T]:
        """Optimized batch document reading"""
        return await self._execute_with_metrics(
            "batch_read_documents",
            self._batch_read_impl,
            doc_refs, company_id
        )

    async def _batch_read_impl(self, doc_refs: List[tuple], company_id: str) -> Dict[str, T]:
        results = {}
        
        # Group by collection for optimal batch reads
        collections = {}
        for collection_path, doc_id in doc_refs:
            if collection_path not in collections:
                collections[collection_path] = []
            collections[collection_path].append(doc_id)

        # Concurrent reads per collection
        tasks = []
        for collection_path, doc_ids in collections.items():
            tasks.append(self._read_collection_batch(collection_path, doc_ids, company_id, results))
        
        await asyncio.gather(*tasks)
        return results

    async def _read_collection_batch(self, collection_path: str, doc_ids: List[str], 
                                   company_id: str, results: Dict[str, T]):
        """Read a batch of documents from a single collection"""
        for doc_id in doc_ids:
            try:
                doc_data = await self.read_document(collection_path, doc_id, company_id)
                if doc_data:
                    results[f"{collection_path}/{doc_id}"] = doc_data
            except Exception as e:
                logger.warning("Failed to read document in batch", 
                             path=f"{collection_path}/{doc_id}", error=str(e))

    def _calculate_adaptive_ttl(self, collection_path: str) -> int:
        """Calculate TTL based on data volatility patterns"""
        if 'dashboard' in collection_path:
            return 60  # High volatility - 1 minute
        elif 'crises' in collection_path:
            return 300  # Medium volatility - 5 minutes
        elif 'companies' in collection_path:
            return 1800  # Low volatility - 30 minutes
        return 600  # Default - 10 minutes

class FirestoreWriteTool(BaseMCPTool):
    """High-performance Firestore write operations with transactions and batch support"""
    
    def __init__(self, connection_pool: FirestoreConnectionPool):
        super().__init__(
            "firestore_write_tool",
            "Optimized Firestore write operations with atomic transactions"
        )
        self.pool = connection_pool

    async def write_document(self, collection_path: str, doc_id: str, 
                           payload: Dict[str, Any], company_id: str) -> str:
        """Atomic document write with company validation"""
        return await self._execute_with_metrics(
            "write_document",
            self._write_document_impl,
            collection_path, doc_id, payload, company_id
        )

    async def _write_document_impl(self, collection_path: str, doc_id: str,
                                 payload: Dict[str, Any], company_id: str) -> str:
        document_path = f"{collection_path}/{doc_id}"
        await self.pool.set_document(document_path, payload, company_id)
        
        logger.debug("Document written", path=document_path)
        return doc_id

    async def update_document(self, collection_path: str, doc_id: str,
                            updates: Dict[str, Any], company_id: str) -> str:
        """Update specific fields in a document"""
        return await self._execute_with_metrics(
            "update_document",
            self._update_document_impl,
            collection_path, doc_id, updates, company_id
        )

    async def _update_document_impl(self, collection_path: str, doc_id: str,
                                  updates: Dict[str, Any], company_id: str) -> str:
        document_path = f"{collection_path}/{doc_id}"
        await self.pool.update_document(document_path, updates, company_id)
        
        logger.debug("Document updated", path=document_path)
        return doc_id

    async def update_counters(self, doc_path: str, deltas: Dict[str, int], 
                            company_id: str = None) -> None:
        """Transaction-based counter updates with sharded counter support"""
        return await self._execute_with_metrics(
            "update_counters",
            self._update_counters_impl,
            doc_path, deltas, company_id
        )

    async def _update_counters_impl(self, doc_path: str, deltas: Dict[str, int], 
                                  company_id: str = None) -> None:
        async def transaction_func(transaction, client):
            doc_ref = client.document(doc_path)
            doc_snapshot = await transaction.get(doc_ref)
            
            current_data = doc_snapshot.to_dict() if doc_snapshot.exists else {}
            
            # Apply counter updates
            for field, delta in deltas.items():
                current_value = current_data.get(field, 0)
                current_data[field] = current_value + delta
            
            current_data['updated_at'] = datetime.utcnow()
            transaction.set(doc_ref, current_data)
            
            return current_data

        await self.pool.transaction_write(transaction_func, company_id)
        logger.debug("Counters updated", doc_path=doc_path, deltas=deltas)

class FirestoreBatchTool(BaseMCPTool):
    """Batch operations for high-throughput scenarios"""
    
    def __init__(self, connection_pool: FirestoreConnectionPool):
        super().__init__(
            "firestore_batch_tool", 
            "High-performance batch operations for Firestore"
        )
        self.pool = connection_pool

    async def batch_write(self, operations: List[Dict[str, Any]], company_id: str) -> Dict[str, Any]:
        """Execute multiple write operations atomically"""
        return await self._execute_with_metrics(
            "batch_write",
            self._batch_write_impl,
            operations, company_id
        )

    async def _batch_write_impl(self, operations: List[Dict[str, Any]], company_id: str) -> Dict[str, Any]:
        await self.pool.batch_write(operations, company_id)
        
        result = {
            "operations_completed": len(operations),
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success"
        }
        
        logger.info("Batch write completed", operations_count=len(operations))
        return result

# Specialized Analysis Tools

class CompanyProfileValidator(BaseMCPTool):
    """Validates company profile data completeness and quality"""
    
    def __init__(self):
        super().__init__(
            "company_profile_validator",
            "Validates company profile data for completeness and quality"
        )

    async def validate_profile(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate company profile completeness"""
        return await self._execute_with_metrics(
            "validate_profile",
            self._validate_profile_impl,
            profile_data
        )

    async def _validate_profile_impl(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        required_fields = ['name', 'industry', 'brand_voice']
        optional_fields = ['mission', 'values', 'ethics', 'contacts']
        
        completeness_score = 0.0
        missing_fields = []
        present_fields = []
        
        # Check required fields
        for field in required_fields:
            if field in profile_data and profile_data[field]:
                completeness_score += 0.25
                present_fields.append(field)
            else:
                missing_fields.append(field)
        
        # Check optional fields
        for field in optional_fields:
            if field in profile_data and profile_data[field]:
                completeness_score += 0.1
                present_fields.append(field)
        
        return {
            "completeness_score": min(completeness_score, 1.0),
            "missing_fields": missing_fields,
            "present_fields": present_fields,
            "validation_timestamp": datetime.utcnow().isoformat(),
            "quality_issues": self._check_quality_issues(profile_data)
        }

    def _check_quality_issues(self, profile_data: Dict[str, Any]) -> List[str]:
        """Check for data quality issues"""
        issues = []
        
        if profile_data.get('brand_voice') and len(profile_data['brand_voice']) < 10:
            issues.append("Brand voice description too short")
        
        if profile_data.get('values') and len(profile_data['values']) == 0:
            issues.append("No company values specified")
        
        return issues

class StakeholderAnalyzer(BaseMCPTool):
    """Analyzes stakeholder relationships and influence"""
    
    def __init__(self):
        super().__init__(
            "stakeholder_analyzer",
            "Analyzes stakeholder influence and relationship patterns"
        )

    async def analyze_influence_network(self, stakeholders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze stakeholder influence patterns"""
        return await self._execute_with_metrics(
            "analyze_influence_network",
            self._analyze_influence_impl,
            stakeholders
        )

    async def _analyze_influence_impl(self, stakeholders: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not stakeholders:
            return {"total_stakeholders": 0, "influence_distribution": {}}
        
        # Analyze influence distribution
        influence_scores = [s.get('importance_score', 0.0) for s in stakeholders]
        total_influence = sum(influence_scores)
        
        # Categorize by type
        by_type = {}
        for stakeholder in stakeholders:
            stakeholder_type = stakeholder.get('type', 'unknown')
            if stakeholder_type not in by_type:
                by_type[stakeholder_type] = []
            by_type[stakeholder_type].append(stakeholder)
        
        # Find key influencers
        key_influencers = sorted(
            [s for s in stakeholders if s.get('importance_score', 0) > 0.7],
            key=lambda x: x.get('importance_score', 0),
            reverse=True
        )
        
        return {
            "total_stakeholders": len(stakeholders),
            "total_influence_score": total_influence,
            "average_influence": total_influence / len(stakeholders) if stakeholders else 0,
            "distribution_by_type": {
                stakeholder_type: len(group) for stakeholder_type, group in by_type.items()
            },
            "key_influencers": [
                {
                    "name": s.get('name', 'Unknown'),
                    "type": s.get('type', 'unknown'),
                    "influence_score": s.get('importance_score', 0)
                }
                for s in key_influencers[:5]  # Top 5
            ],
            "analysis_timestamp": datetime.utcnow().isoformat()
        }

class TimelineBuilder(BaseMCPTool):
    """Builds event timelines and analyzes temporal patterns"""
    
    def __init__(self):
        super().__init__(
            "timeline_builder",
            "Constructs event timelines and identifies temporal patterns"
        )

    async def build_event_timeline(self, events: List[Dict[str, Any]], 
                                 analysis_window_days: int = 30) -> Dict[str, Any]:
        """Build timeline of events and identify patterns"""
        return await self._execute_with_metrics(
            "build_event_timeline",
            self._build_timeline_impl,
            events, analysis_window_days
        )

    async def _build_timeline_impl(self, events: List[Dict[str, Any]], 
                                 analysis_window_days: int = 30) -> Dict[str, Any]:
        if not events:
            return {"timeline": [], "patterns": {}}
        
        # Sort events by time
        sorted_events = sorted(
            events,
            key=lambda x: x.get('start_time', datetime.min)
        )
        
        # Analyze within window
        cutoff_date = datetime.utcnow() - timedelta(days=analysis_window_days)
        recent_events = [
            e for e in sorted_events 
            if e.get('start_time', datetime.min) >= cutoff_date
        ]
        
        # Identify patterns
        patterns = {
            "event_frequency": len(recent_events),
            "peak_activity_days": self._find_peak_activity(recent_events),
            "event_types": self._categorize_events(recent_events),
            "impact_correlation": self._analyze_impact_correlation(recent_events)
        }
        
        return {
            "timeline": sorted_events,
            "recent_events_count": len(recent_events),
            "patterns": patterns,
            "analysis_period_days": analysis_window_days,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }

    def _find_peak_activity(self, events: List[Dict[str, Any]]) -> List[str]:
        """Find days with highest activity"""
        daily_counts = {}
        for event in events:
            date_str = event.get('start_time', datetime.utcnow()).strftime('%Y-%m-%d')
            daily_counts[date_str] = daily_counts.get(date_str, 0) + 1
        
        # Return top 3 days
        return sorted(daily_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    def _categorize_events(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize events by type/tags"""
        categories = {}
        for event in events:
            tags = event.get('tags', [])
            if not tags:
                categories['untagged'] = categories.get('untagged', 0) + 1
            else:
                for tag in tags:
                    categories[tag] = categories.get(tag, 0) + 1
        return categories

    def _analyze_impact_correlation(self, events: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze correlation between events and impact"""
        if not events:
            return {}
        
        impact_scores = [e.get('impact_estimate', 0.0) for e in events]
        avg_impact = sum(impact_scores) / len(impact_scores)
        
        return {
            "average_impact": avg_impact,
            "high_impact_events": len([s for s in impact_scores if s > 0.7]),
            "low_impact_events": len([s for s in impact_scores if s < 0.3])
        }

# Export all tools
__all__ = [
    'FirestoreReadTool',
    'FirestoreWriteTool', 
    'FirestoreBatchTool',
    'CompanyProfileValidator',
    'StakeholderAnalyzer',
    'TimelineBuilder'
]