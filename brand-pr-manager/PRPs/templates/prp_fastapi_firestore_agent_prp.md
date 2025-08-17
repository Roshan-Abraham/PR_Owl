---
name: "FastAPI + Firestore + MCP Agents PRP"
description: "Template PRP for planning a FastAPI application using Firestore as primary DB, MCP/ADK agents, and a Vector DB for case-study review and recommendations."
---

## Purpose

This PRP describes a production-ready template for building a crisis simulation and response system:

- FastAPI main API service to run simulations and orchestrate agent workflows
- Google Cloud Firestore as the canonical application datastore
- ADK (Agent Development Kit) agents deployed to Vertex AI Agent Engine; use ADK primitives (Workflow agents, LlmAgent, AgentTool) to structure parent/sub-agent flows. Vertex AI (Gemini-family) is the LLM provider.
- Vector DB: Milvus as the single vector store for indexed case studies and retrieval-augmented reasoning
- Memory and session management: Vertex AI Memory Bank (via ADK's VertexAiMemoryBankService) for long-term, semantic memory; see memory scoping notes below
- Messaging: Pub/Sub is optional — ADK sessions + Memory Bank provide multi-turn persistence and agent coordination; include Pub/Sub when durable cross-service eventing, external triggers, or long-lived background jobs are required

Use this PRP to generate a concrete project scaffold, diagrams, and checklists suitable for engineers to implement the system.

## Requirements extracted (checklist)

- [ ] FastAPI service for running simulations and exposing snapshot/classify/recommend endpoints
- [ ] Firestore data model for multi-company support and crisis-case-centric tracking
- [ ] Agent architecture: context-collector, classifier (+ sub-agents), recommender (uses Vector DB)
- [ ] All agent outputs persisted to Firestore
- [ ] Firestore CRUD tool and access/query rules for agents
- [ ] Recommended messaging/agent-communication architecture
- [ ] Vector DB integration strategy and link to docs
- [ ] Security considerations and Firestore rules examples
- [ ] Success criteria and validation loops

## High-level architecture

1. FastAPI service receives a simulation request (POST /simulate). It creates a new CrisisCase document in Firestore and publishes a "crisis.created" event.
2. Context Collector Agent (Agent A) subscribes to events (or is invoked via MCP tool). It gathers company profile, recent events, relations, and other context from Firestore and writes a CrisisSnapshot document. End of Agent A.
3. Classification Agent (Agent B) runs, using the snapshot to:
   - classify crisis severity and class (e.g., Product, Legal, Social, Financial)
   - produce a scorecard (multiple metrics)
   - run sub-agents to detect affected entities, relations, upcoming events and compute impact scores
     It writes results to the CrisisCase.scorecard and related subcollections.
4. Recommendation Agent (Agent C) queries the Vector DB (case-study index) using embeddings of the crisis snapshot, scorecard, and entity contexts. It generates a step-by-step engagement and resolution plan. Writes recommendations into Firestore and updates the CrisisCase.status.
5. Observability & dashboard service reads the dashboard collection (summary) and surfaces updates.

Recommended communication: publish/subscribe for eventing (Google Pub/Sub or Redis Streams). Agents may also be directly invoked through MCP calls where a central orchestrator triggers each agent step.

## Why Pub/Sub / Messaging

- decouples agents from API layer
- enables retries and durable processing
- supports fan-out to sub-agents (parallel classification tasks)

Docs:

- FastAPI: https://fastapi.tiangolo.com/
- Firestore docs (overview & data modelling): https://cloud.google.com/firestore/docs/overview
- Firebase Admin setup (server SDKs): https://firebase.google.com/docs/admin/setup
- Firestore Security Rules: https://firebase.google.com/docs/rules
- Vertex AI Agent Engine & ADK: https://google.github.io/adk-docs/deploy/agent-engine/
- ADK Memory docs / Vertex AI Memory Bank: https://google.github.io/adk-docs/sessions/memory/
- ADK multi-agent patterns: https://google.github.io/adk-docs/agents/multi-agents/
- Milvus (Vector DB): https://milvus.io/docs/overview.md
- Milvus python client & standalone install: https://milvus.io/docs/install_standalone-docker.md

## FastAPI endpoints (suggested)

- POST /simulate
  - Creates CrisisCase, optionally receives template id and simulation inputs
  - Returns crisis_case_id
- GET /crisis/{crisis_case_id}/snapshot
  - Returns the latest CrisisSnapshot document
- GET /crisis/{crisis_case_id}
  - Full crisis case document with scorecard and recommendations
- POST /crisis/{crisis_case_id}/classify (optional manual trigger)
- POST /crisis/{crisis_case_id}/recommend (optional manual trigger)
- CRUD endpoints for company profiles, events, templates, and vector DB ingestion

## Enhanced Agent Architecture with DB Sub-agents

### Enhanced Context Collector (Agent A) + Specialized Sub-agents

**Main Orchestrator**: Context Collector Agent

- **Input**: `{ crisis_case_id, company_id, session_id, origin_point }`
- **Output**: Complete CrisisSnapshot with validated context data
- **Coordination**: Uses ADK SequentialAgent to orchestrate 7 specialized sub-agents
- **Memory Integration**: Session-scoped context accumulation in Vertex AI Memory Bank

**Diversified Sub-agents with MCP Tool Integration**:

1. **CompanyProfileAgent** (Company Core Data Specialist)

   - **MCP Tools**: `FirestoreReadTool`, `CompanyProfileValidator`
   - **Specialty**: Company profile, brand voice, industry context, settings validation
   - **Query patterns**: `companies/{company_id}` + `companies/{company_id}/details/profile`
   - **Output**: Validated company context with completeness score

2. **StakeholderMappingAgent** (Relationship Intelligence Specialist)

   - **MCP Tools**: `FirestoreReadTool`, `StakeholderAnalyzer`, `RelationshipGraphBuilder`
   - **Specialty**: Stakeholder influence mapping, communication history analysis, sentiment tracking
   - **Query patterns**: `companies/{company_id}/relations` with importance scoring, communication history analysis
   - **Output**: Stakeholder influence graph with communication preferences

3. **EventContextAgent** (Temporal Context Specialist)

   - **MCP Tools**: `FirestoreReadTool`, `TimelineBuilder`, `EventImpactAnalyzer`
   - **Specialty**: Recent events correlation, timeline reconstruction, event impact prediction
   - **Query patterns**: Date-filtered events, overlapping timelines, impact cascade analysis
   - **Output**: Contextual timeline with event interdependencies

4. **HistoricalPatternAgent** (Crisis History Specialist)

   - **MCP Tools**: `FirestoreReadTool`, `PatternRecognizer`, `CrisisClassifier`
   - **Specialty**: Historical crisis patterns, response effectiveness, outcome prediction
   - **Query patterns**: Historical snapshots, outcome tracking, pattern analysis
   - **Output**: Historical context with pattern insights and precedent cases

5. **ExternalSignalsAgent** (External Intelligence Specialist)

   - **MCP Tools**: `WebScrapingTool`, `SocialMediaAnalyzer`, `NewsAggregator`
   - **Specialty**: External market signals, news analysis, social sentiment, industry trends
   - **Data Sources**: News APIs, social media monitoring, industry reports
   - **Output**: External context signals with credibility scoring

6. **KnowledgeBaseAgent** (Internal Knowledge Specialist)

   - **MCP Tools**: `FirestoreReadTool`, `VectorSearchTool`, `DocumentSummarizer`
   - **Specialty**: Company knowledge base, internal case studies, lessons learned
   - **Query patterns**: `companies/{company_id}/knowledge_base`, vector similarity search
   - **Output**: Relevant internal knowledge with applicability scoring

7. **SnapshotSynthesizer** (Data Integration Specialist)
   - **MCP Tools**: `FirestoreWriteTool`, `DataValidator`, `ContextAggregator`
   - **Specialty**: Context synthesis, data validation, snapshot generation, quality assurance
   - **Operations**: Multi-source data fusion, completeness validation, atomic snapshot creation
   - **Output**: Validated CrisisSnapshot with confidence metrics and data lineage

### Enhanced Classification Agent (Agent B) + Multi-Dimensional Analysis Sub-agents

**Main Orchestrator**: Classification Agent

- **Input**: `{ snapshot_id, crisis_case_id, company_id, session_id }`
- **Output**: Comprehensive scorecard with multi-dimensional risk assessment
- **Coordination**: Uses ADK ParallelAgent for concurrent analysis + SequentialAgent for synthesis
- **Memory Integration**: Classification patterns and precedents stored in Memory Bank

**Specialized Analysis Sub-agents**:

1. **SeverityAssessmentAgent** (Crisis Magnitude Specialist)

   - **MCP Tools**: `FirestoreReadTool`, `SeverityCalculator`, `BenchmarkAnalyzer`
   - **Specialty**: Crisis magnitude assessment, industry benchmarking, severity thresholds
   - **Analysis Dimensions**: Scale, urgency, complexity, cascading potential
   - **Output**: Multi-dimensional severity scores with confidence intervals

2. **ImpactPredictionAgent** (Consequence Analysis Specialist)

   - **MCP Tools**: `ImpactModeler`, `ScenarioGenerator`, `CascadeAnalyzer`
   - **Specialty**: Impact prediction modeling, cascade analysis, outcome scenarios
   - **Analysis Dimensions**: Financial, reputational, operational, regulatory, market impact
   - **Output**: Impact prediction matrix with probability distributions

3. **StakeholderExposureAgent** (Stakeholder Risk Specialist)

   - **MCP Tools**: `FirestoreReadTool`, `ExposureCalculator`, `StakeholderMapper`
   - **Specialty**: Stakeholder exposure analysis, communication risk, relationship impact
   - **Analysis Dimensions**: Customer impact, investor exposure, partner effects, regulatory attention
   - **Output**: Stakeholder exposure matrix with communication priorities

4. **TimelineAnalysisAgent** (Temporal Dynamics Specialist)

   - **MCP Tools**: `TemporalAnalyzer`, `UrgencyCalculator`, `WindowOptimizer`
   - **Specialty**: Time-sensitive analysis, response windows, escalation timing
   - **Analysis Dimensions**: Response urgency, escalation triggers, optimal timing windows
   - **Output**: Temporal risk assessment with action timing recommendations

5. **CompetitiveContextAgent** (Market Position Specialist)

   - **MCP Tools**: `CompetitorAnalyzer`, `MarketPositionTool`, `BrandImpactAssessor`
   - **Specialty**: Competitive implications, market positioning impact, brand differentiation risk
   - **Analysis Dimensions**: Competitive advantage impact, market share risk, brand positioning
   - **Output**: Competitive risk analysis with positioning recommendations

6. **LegalComplianceAgent** (Regulatory Risk Specialist)

   - **MCP Tools**: `RegulatoryAnalyzer`, `ComplianceChecker`, `LegalRiskAssessor`
   - **Specialty**: Legal implications, regulatory compliance, litigation risk
   - **Analysis Dimensions**: Regulatory violations, litigation exposure, compliance gaps
   - **Output**: Legal risk assessment with compliance recommendations

7. **ScorecardSynthesizerAgent** (Integration and Validation Specialist)
   - **MCP Tools**: `FirestoreWriteTool`, `ScoreAggregator`, `ValidationEngine`, `DashboardUpdater`
   - **Specialty**: Multi-dimensional score integration, validation, dashboard updates
   - **Operations**: Score aggregation, confidence calculation, atomic scorecard creation, dashboard metrics update
   - **Output**: Validated comprehensive scorecard with aggregated confidence metrics

### Enhanced Recommendation Agent (Agent C) + Strategic Planning Sub-agents

**Main Orchestrator**: Recommendation Agent

- **Input**: `{ crisis_case_id, scorecard_id, snapshot_id, company_id, session_id }`
- **Output**: Comprehensive strategic response plan with multiple scenarios and actionable steps
- **Coordination**: Uses ADK WorkflowAgent for complex multi-step recommendation generation
- **Memory Integration**: Strategic patterns and successful responses stored in Memory Bank with company-scoped access

**Strategic Planning Sub-agents**:

1. **HistoricalCaseSearchAgent** (Precedent Research Specialist)

   - **MCP Tools**: `VectorSearchTool`, `CaseStudyAnalyzer`, `SimilarityRanker`
   - **Specialty**: Historical case retrieval, precedent analysis, outcome correlation
   - **Search Strategy**: Multi-vector search (context, severity, industry, stakeholders), metadata filtering
   - **Output**: Ranked list of similar cases with applicability scores and outcome analysis

2. **ScenarioModelingAgent** (Strategic Options Specialist)

   - **MCP Tools**: `ScenarioGenerator`, `OptionAnalyzer`, `RiskModeler`
   - **Specialty**: Multiple response scenarios, option analysis, strategy branching
   - **Strategy Types**: Defensive, proactive, collaborative, competitive response strategies
   - **Output**: Multiple strategic scenarios with trade-off analysis

3. **StakeholderStrategyAgent** (Communication Planning Specialist)

   - **MCP Tools**: `FirestoreReadTool`, `CommunicationPlanner`, `MessageCrafter`
   - **Specialty**: Stakeholder-specific communication strategies, message timing, channel selection
   - **Communication Dimensions**: Internal, customer, partner, regulatory, media communications
   - **Output**: Stakeholder communication matrix with timing and channel recommendations

4. **ResourceOptimizationAgent** (Implementation Planning Specialist)

   - **MCP Tools**: `ResourceCalculator`, `TimelineOptimizer`, `CapacityPlanner`
   - **Specialty**: Resource allocation, timeline optimization, capacity planning, budget estimation
   - **Optimization Dimensions**: Human resources, financial investment, time allocation, external services
   - **Output**: Resource allocation plan with timeline and budget estimates

5. **RiskMitigationAgent** (Contingency Planning Specialist)

   - **MCP Tools**: `RiskAssessor`, `ContingencyPlanner`, `FallbackGenerator`
   - **Specialty**: Risk mitigation strategies, contingency planning, fallback options
   - **Mitigation Areas**: Escalation prevention, damage control, recovery strategies
   - **Output**: Comprehensive risk mitigation plan with contingency triggers

6. **ComplianceValidatorAgent** (Legal and Regulatory Specialist)

   - **MCP Tools**: `ComplianceChecker`, `RegulatoryAnalyzer`, `LegalValidator`
   - **Specialty**: Legal compliance validation, regulatory requirements, policy adherence
   - **Validation Areas**: Industry regulations, internal policies, legal constraints
   - **Output**: Compliance validation report with regulatory considerations

7. **RecommendationSynthesizerAgent** (Strategic Integration Specialist)
   - **MCP Tools**: `FirestoreWriteTool`, `PlanIntegrator`, `ConfidenceCalculator`, `AuditLogger`
   - **Specialty**: Strategy integration, plan synthesis, confidence assessment, recommendation finalization
   - **Operations**: Multi-scenario integration, confidence scoring, atomic recommendation creation, audit trail
   - **Output**: Integrated strategic recommendation with execution roadmap and confidence metrics

**Session-Scoped Agent Coordination**:

```python
crisis_session = {
    "session_id": f"{company_id}:{crisis_id}",
    "agents": {
        "context_collector": {
            "readers": ["CompanyDataAgent", "HistoryReaderAgent", "EventsReaderAgent", "RelationsReaderAgent"],
            "writers": ["SnapshotWriterAgent"]
        },
        "classifier": {
            "readers": ["SnapshotReaderAgent"],
            "writers": ["ScorecardWriterAgent", "DashboardUpdaterAgent"],
            "analyzers": ["SeverityAnalyzerAgent", "EntityDetectorAgent", "ImpactScorerAgent"]
        },
        "recommender": {
            "readers": ["CaseDataReaderAgent"],
            "searchers": ["VectorSearchAgent"],
            "generators": ["StrategyGeneratorAgent", "CostEstimatorAgent"],
            "writers": ["RecommendationWriterAgent", "AuditLoggerAgent"]
        }
    }
}
```

All sub-agent outputs must be persisted to Firestore with proper session scoping and emit ADK events for coordination.

## Enhanced Firestore Schema with Complete Data Model

**Top-level collections (multitenant-aware):**

- **companies/{company_id}**

  - document fields: `{id: company_id, name, timezone, industry, brand_voice, created_at, settings: {notification_preferences, escalation_thresholds, ai_model_preferences}, metadata: {subscription_tier, feature_flags}}`
  - subcollections:
    - **dashboard** -> `summary` document `{ company_id, num_active_crises, num_critical, num_resolved_24h, avg_resolution_time_hours, last_updated, trend_data: {severity_trend, volume_trend} }`
    - **details** -> `profile` document `{ mission, values: [str], ethics: [str], bio, contacts: [{name, role, email, phone}], key_stakeholders: [{name, influence_score, contact_method}] }`
    - **events** -> event documents `{ event_id, title, description, tags:[], demographics: {audience_size, key_groups}, start_time, end_time, impact_estimate, stakeholder_involvement, media_coverage_expected }`
    - **relations** -> relation documents `{ relation_id, name, type: "customer|partner|investor|regulator|media", importance_score: 0-1, contact_info, communication_history, sentiment_history }`
    - **templates** -> crisis simulation templates `{ template_id, name, scenario_type, complexity_level, learning_objectives, default_parameters, estimated_duration }`
    - **knowledge_base** -> company-specific case studies `{ kb_id, title, category, lessons_learned, outcome_summary, vector_ids: [str], tags }`

- **Company/{company_id}/Crises/{crisis_case_id}**

  - fields: `{
  id: crisis_case_id,
  session_id: "{company_id}:{crisis_id}",
  created_at,
  updated_at,
  origin_point: {type: "simulation|real|external", source: "template_id|news_url|social_post_id", metadata: {}},
  nature: "product_defect|regulatory|social|financial|operational|legal",
  current_status: "created|context_collected|classified|recommendation_generated|action_planned|resolved|archived",
  primary_class,
  severity_score: 0-1,
  confidence_score: 0-1,
  affected_stakeholders: [str],
  estimated_resolution_time_hours,
  latest_snapshot_id,  # points to an Artifact id
  latest_scorecard_id, # points to an Artifact id
  latest_recommendation_id, # points to an Artifact id
  summary: "brief_description_for_dashboard"
  }`
  - subcollections:
    - **Artifacts/{artifact_id}** -> `{ artifact_id, artifact_type: "snapshot|scorecard|recommendation|other", created_at, origin_metadata, payload: {...} }`
    - **logs/{log_id}** -> `{ log_id, timestamp, agent_id, sub_agent_id, action, input_data, output_data, execution_time_ms, status, error_details }`
    - **agent_sessions/{session_id}** -> `{ session_id, created_at, agent_coordination_state, memory_bank_references, current_step, completed_steps }`

- **agent_runs/{run_id}** (global agent execution tracking)

  - fields: `{ run_id, company_id, crisis_case_id, agent_type, sub_agents_involved: [str], start_timestamp, end_timestamp, status, error_details, performance_metrics: {execution_time_ms, memory_used, tokens_consumed} }`

- **vector_metadata/{collection_name}/objects/{object_id}** (Milvus metadata sync)

  - fields: `{ object_id, milvus_collection, vector_id, source_type: "case_study|company_knowledge|external", source_id, company_id, tags, embeddings_model, created_at, last_updated }`

- **dashboards/{company_id}** (top-level for fast access)
  - fields: `{ company_id, summary: {num_crises_total, num_active, num_critical, num_resolved_24h, avg_resolution_time_hours}, trend_data: {severity_trend_7d, volume_trend_7d}, last_updated, alert_thresholds }`

Production-ready additions and recommendations

- Partitioning & multi-tenancy: Keep `company_id` as the primary logical partition key. For queries that are per-company, always scope queries with `where('company_id','==', company_id)` to avoid collection scans and to make security rules simple.
- Collections vs subcollections: keep hot-access small documents at top-level (e.g., `crises/{id}` with metadata and pointers) and place large or historical payloads in subcollections (e.g., `Artifacts`, `logs`, `history`) to avoid document size limits and hot-document contention.
- Per-company dashboard/summary: maintain `dashboards/{company_id}` top-level collection with a single `summary` document for fast reads. Use transactions to update counters (avoid read-modify-write races).
- Deterministic IDs & idempotency: generate deterministic IDs when re-runs are possible (e.g., `crisis_{uuid}` or `snapshot::{source_id}`) so writes are idempotent; use `create_time` and `update_time` fields.
- Composite indexes: plan composite indexes for the common query patterns:
  - query by company + status + severity: `(company_id, current_status, severity_score desc)`
  - query by company + updated_at range for dashboard: `(company_id, updated_at desc)`
  - query by company + tags: `(company_id, tags)` (use array-contains or maintain tag counters)
- Query patterns to support:
  - list crises for company sorted by severity: where(company_id==X).order_by('severity_score', desc=True).limit(50)
    - fetch recent snapshots by crisis: collection('Company').document(company_id).collection('Crises').document(id).collection('Artifacts').where('artifact_type','==','snapshot').order_by('payload.created_at', desc=True).limit(10)
  - search ai_insights for company with tag filters: where(company_id==X).where('tags','array_contains_any', ['safety','legal'])
- Denormalization: duplicate small summary fields (e.g., `latest_recommendation_summary`) on the `crises` document so dashboards can be served without joins.
- Auditing & immutable logs: store immutable events in `crises/{id}/logs` with fields `{author, action, payload, timestamp}`. Use these for reproducing agent runs.
- Sharding and write scaling: if a particular field becomes a hotspot (e.g., counter updates on dashboard summary), use a sharded counter pattern or move write-heavy metrics to Bigtable / BigQuery for analytics.

Indexes and capacity planning

- Monitor query patterns with Firestore usage dashboards and create composite indexes ahead of release for the queries above.
- Document size: keep documents < 1 MB (Firestore hard limit). Push large blobs into object storage (GCS) and store pointers in Firestore metadata.

Security & path-based write rules for agents

- Agents should write using server-side Admin SDK with service account credentials; client UI uses strict Firestore rules.
- Provide agents helper methods that resolve write paths based on the case type and agent role (see `firestore_client.py` helper below). Agents must never construct raw paths in UIs; use server helpers which enforce canonical paths and field masks for writes.

- ai_insights/{company_id}/cases/{case_id}

  - fields: {impact_estimate, contributors, actions_recommended, created_at}

- simulations/{company_id}/templates/{template_id}
  - fields: {name, description, learning_objectives, params}
- simulations/{company_id}/history/{history_id}

  - fields: {template_id, crisis_case_id, run_results, score}

- vectordb_metadata/{namespace}/objects/{object_id}
  - metadata for vectors (case study id, source, date, tags, company_id optional)

Indexing & IDs

- Use Firestore auto-IDs for most collections, but generate stable IDs for crises using UUIDv4 with a prefix like `crisis_{uuid}` so clients can correlate.
- Keep company_id as canonical partition key.

Dashboard collection

- For quick dashboard reads maintain `dashboards/{company_id}` summary document with aggregates: { num_crises_total, num_critical, last_updated, avg_resolution_time }
- Write/update this document within agent flows (atomic where possible using transactions).

## Example Firestore document example (CrisisCase)

{
"id": "crisis_9f3a7b...",
"company_id": "company_acme",
"created_at": "2025-08-15T12:00:00Z",
"origin_point": { "type": "simulation", "source": "template_xyz" },
"nature": "social",
"current_status": "classified",
"primary_class": "social_media_backlash",
"severity_score": 0.78
}

## Enhanced Firestore CRUD + MCP Tools Integration

**Purpose**: Provide FastMap-optimized library with MCP tool integration for agents, featuring connection pooling, intelligent caching, type-safe queries, and batch operations.

### MCP Tools for DB Sub-agents

**FastMap-Optimized FirestoreReadTool**:

```python
from google.adk import AgentTool
from typing import Dict, List, Optional, Any, TypeVar, Generic
from dataclasses import dataclass
from cachetools import TTLCache
import asyncio

T = TypeVar('T')

@dataclass
class QueryFilter:
    field: str
    operator: str
    value: Any

@dataclass
class QueryOptions:
    limit: Optional[int] = None
    offset: Optional[int] = None
    order_by: Optional[str] = None
    order_direction: str = "asc"

class FirestoreReadTool(AgentTool, Generic[T]):
    def __init__(self, connection_pool, cache_manager: TTLCache, schema_validator):
        self.pool = connection_pool
        self.cache = cache_manager
        self.validator = schema_validator
        self.query_plan_cache = TTLCache(maxsize=500, ttl=900)  # 15-minute query plan cache

    async def read_document(self, collection_path: str, doc_id: str, company_id: str,
                          expected_type: Optional[type] = None) -> Optional[T]:
        """Company-scoped document read with type safety and intelligent caching"""
        cache_key = f"{company_id}:{collection_path}:{doc_id}"

        # Check cache first
        if cached_result := self.cache.get(cache_key):
            return cached_result

        # Validate company scope
        if not self._validate_company_access(collection_path, company_id):
            raise SecurityError(f"Unauthorized access to {collection_path} for company {company_id}")

        async with self.pool.get_connection() as conn:
            doc_ref = conn.collection(collection_path).document(doc_id)
            doc_snapshot = await doc_ref.get()

            if not doc_snapshot.exists:
                return None

            data = doc_snapshot.to_dict()

            # Type validation
            if expected_type and not self.validator.validate(data, expected_type):
                raise ValidationError(f"Document {doc_id} doesn't match expected schema")

            # Cache with adaptive TTL based on data volatility
            ttl = self._calculate_adaptive_ttl(collection_path)
            self.cache.set(cache_key, data, ttl=ttl)

            return data

    async def query_collection(self, collection_path: str, filters: List[QueryFilter],
                             company_id: str, options: QueryOptions = None) -> List[T]:
        """High-performance collection queries with compound index optimization"""
        query_signature = self._generate_query_signature(collection_path, filters, options)

        # Check query plan cache
        if cached_plan := self.query_plan_cache.get(query_signature):
            execution_plan = cached_plan
        else:
            execution_plan = self._optimize_query_plan(filters, options)
            self.query_plan_cache[query_signature] = execution_plan

        async with self.pool.get_connection() as conn:
            query = conn.collection(collection_path)

            # Always scope by company_id first for security and performance
            query = query.where("company_id", "==", company_id)

            # Apply optimized filters based on execution plan
            for filter_group in execution_plan.filter_groups:
                for f in filter_group:
                    query = query.where(f.field, f.operator, f.value)

            # Apply ordering and limits
            if options and options.order_by:
                query = query.order_by(options.order_by, direction=options.order_direction)
            if options and options.limit:
                query = query.limit(options.limit)

            results = []
            async for doc in query.stream():
                data = doc.to_dict()
                results.append(data)

            return results

    async def batch_read_documents(self, doc_refs: List[tuple[str, str]], company_id: str) -> Dict[str, T]:
        """Optimized batch document reading with connection pooling"""
        results = {}

        # Group by collection for optimal batch reads
        collections = {}
        for collection_path, doc_id in doc_refs:
            if collection_path not in collections:
                collections[collection_path] = []
            collections[collection_path].append(doc_id)

        # Concurrent batch reads per collection
        async def read_collection_batch(collection_path: str, doc_ids: List[str]):
            async with self.pool.get_connection() as conn:
                docs = await conn.collection(collection_path).select(doc_ids).get()
                for doc in docs:
                    if doc.exists and self._validate_company_access_data(doc.to_dict(), company_id):
                        results[f"{collection_path}/{doc.id}"] = doc.to_dict()

        await asyncio.gather(*[
            read_collection_batch(path, doc_ids)
            for path, doc_ids in collections.items()
        ])

        return results

    def _calculate_adaptive_ttl(self, collection_path: str) -> int:
        """Calculate TTL based on data volatility patterns"""
        if 'dashboard' in collection_path:
            return 60  # High volatility - 1 minute
        elif 'crises' in collection_path:
            return 300  # Medium volatility - 5 minutes
        elif 'companies' in collection_path:
            return 1800  # Low volatility - 30 minutes
        return 600  # Default - 10 minutes
```

**FirestoreWriteTool**:

```python
class FirestoreWriteTool(AgentTool):
    async def write_document(self, collection_path: str, doc_id: str, payload: Dict, company_id: str) -> str:
        # Atomic writes with optimistic locking
        # Schema validation before write
        # Audit logging integration
        # Retry logic with exponential backoff
        pass

    async def update_counters(self, doc_path: str, deltas: Dict[str, int]) -> None:
        # Transaction-based counter updates
        # Sharded counter support for high-volume
        pass
```

**FirestoreBatchTool**:

```python
class FirestoreBatchTool(AgentTool):
    async def batch_write(self, operations: List[WriteOperation], company_id: str) -> BatchResult:
        # Batched writes with partial failure handling
        # Cross-collection atomicity where needed
        # Progress tracking for long operations
        pass
```

**Enhanced VectorSearchTool with FastMap Optimization**:

```python
class VectorSearchTool(AgentTool):
    def __init__(self, milvus_client, embedding_service, cache_manager):
        self.milvus = milvus_client
        self.embedding_service = embedding_service
        self.cache = cache_manager
        self.search_plan_cache = TTLCache(maxsize=200, ttl=1800)

    async def similarity_search(self, query_text: str, filters: Dict, company_id: str,
                              top_k: int = 10, search_params: Dict = None) -> List[SearchResult]:
        """Company-scoped vector similarity search with intelligent caching"""

        # Generate or retrieve cached embeddings
        embedding_cache_key = f"emb:{hash(query_text)}"
        if cached_embedding := self.cache.get(embedding_cache_key):
            query_vector = cached_embedding
        else:
            query_vector = await self.embedding_service.embed_text(query_text)
            self.cache.set(embedding_cache_key, query_vector, ttl=3600)

        return await self.vector_similarity_search(query_vector, filters, company_id, top_k, search_params)

    async def vector_similarity_search(self, query_vector: List[float], filters: Dict,
                                     company_id: str, top_k: int = 10,
                                     search_params: Dict = None) -> List[SearchResult]:
        """Direct vector search with company partitioning and metadata filtering"""

        # Company-scoped collection selection
        collection_name = f"crisis_vectors_{company_id}" if filters.get('company_scoped') else 'global_crisis_vectors'

        search_params = search_params or {
            "metric_type": "COSINE",
            "params": {"nprobe": min(16, top_k * 2)}
        }

        # Build metadata filters
        expr_parts = [f'company_id == "{company_id}"'] if not filters.get('company_scoped') else []

        if source_types := filters.get('source_types'):
            source_filter = ' or '.join([f'source_type == "{st}"' for st in source_types])
            expr_parts.append(f'({source_filter})')

        if date_range := filters.get('date_range'):
            expr_parts.append(f'created_at >= {date_range["start"]} and created_at <= {date_range["end"]}')

        if tags := filters.get('tags'):
            tag_filter = ' or '.join([f'tags like "%{tag}%"' for tag in tags])
            expr_parts.append(f'({tag_filter})')

        filter_expression = ' and '.join(expr_parts) if expr_parts else None

        # Execute search with retry logic
        try:
            results = await self.milvus.search(
                collection_name=collection_name,
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=top_k * 2,  # Over-fetch for diversity
                expr=filter_expression,
                output_fields=["source_id", "source_type", "title", "summary", "tags", "confidence_score", "created_at"]
            )

            # Post-process results for diversity and relevance
            processed_results = self._diversify_results(results[0], top_k)

            return [
                SearchResult(
                    id=hit.id,
                    similarity_score=hit.score,
                    metadata={field: hit.entity.get(field) for field in hit.entity.fields},
                    source_id=hit.entity.get("source_id"),
                    source_type=hit.entity.get("source_type"),
                    title=hit.entity.get("title"),
                    summary=hit.entity.get("summary")
                ) for hit in processed_results
            ]

        except Exception as e:
            # Fallback to simplified search
            logger.warning(f"Vector search failed, using fallback: {e}")
            return await self._fallback_search(query_vector, company_id, top_k)

    async def hybrid_search(self, query_text: str, keyword_filters: Dict,
                          vector_filters: Dict, company_id: str, top_k: int = 10) -> List[SearchResult]:
        """Hybrid search combining vector similarity and keyword matching"""

        # Parallel vector and keyword searches
        vector_task = self.similarity_search(query_text, vector_filters, company_id, top_k)
        keyword_task = self._keyword_search(query_text, keyword_filters, company_id, top_k)

        vector_results, keyword_results = await asyncio.gather(vector_task, keyword_task)

        # Combine and re-rank results
        combined_results = self._combine_search_results(vector_results, keyword_results, top_k)

        return combined_results

    async def batch_similarity_search(self, queries: List[str], company_id: str,
                                    top_k: int = 5) -> Dict[str, List[SearchResult]]:
        """Optimized batch search for multiple queries"""

        # Batch embedding generation
        embeddings = await self.embedding_service.embed_texts(queries)

        # Concurrent vector searches
        search_tasks = [
            self.vector_similarity_search(emb, {"company_scoped": True}, company_id, top_k)
            for emb in embeddings
        ]

        results = await asyncio.gather(*search_tasks)

        return {query: result for query, result in zip(queries, results)}

    def _diversify_results(self, search_results: List, target_count: int) -> List:
        """Ensure result diversity by source type and recency"""
        if len(search_results) <= target_count:
            return search_results

        # Group by source type for diversity
        by_source = {}
        for result in search_results:
            source_type = result.entity.get('source_type', 'unknown')
            if source_type not in by_source:
                by_source[source_type] = []
            by_source[source_type].append(result)

        # Select diverse results
        diversified = []
        source_types = list(by_source.keys())
        type_index = 0

        while len(diversified) < target_count and any(by_source.values()):
            current_type = source_types[type_index % len(source_types)]
            if by_source[current_type]:
                diversified.append(by_source[current_type].pop(0))
            type_index += 1

        return diversified[:target_count]

@dataclass
class SearchResult:
    id: str
    similarity_score: float
    metadata: Dict[str, Any]
    source_id: str
    source_type: str
    title: str
    summary: str
```

### Enhanced CRUD Helper Functions

**Type-safe, session-scoped operations**:

```python
from typing import TypedDict, Optional
from dataclasses import dataclass

@dataclass
class CrisisSession:
    session_id: str
    company_id: str
    crisis_id: str
    created_at: datetime

class CompanyProfile(TypedDict):
    id: str
    name: str
    timezone: str
    industry: str
    settings: Dict[str, Any]

# Enhanced helper functions with session context
async def get_company_with_session(company_id: str, session: CrisisSession) -> CompanyProfile:
    """Company-scoped read with session validation"""
    tool = FirestoreReadTool()
    return await tool.read_document(f"companies/{company_id}", company_id, session.company_id)

async def create_crisis_with_session(company_id: str, payload: Dict, session: CrisisSession) -> str:
    """Session-scoped crisis creation with audit trail"""
    crisis_id = f"crisis_{uuid4().hex}"
    tool = FirestoreWriteTool()

    # Add session metadata
    payload.update({
        'id': crisis_id,
        'session_id': session.session_id,
        'created_by_agent': True,
        'created_at': firestore.SERVER_TIMESTAMP
    })

    await tool.write_document(f"Company/{company_id}/Crises/{crisis_id}", crisis_id, payload, company_id)
    return crisis_id

async def add_snapshot_with_session(crisis_case_id: str, snapshot: Dict, session: CrisisSession) -> str:
    """Snapshot creation with session scoping and caching"""
    snapshot_id = f"snapshot_{int(time.time())}_{uuid4().hex[:8]}"
    tool = FirestoreWriteTool()

    snapshot.update({
        'id': snapshot_id,
        'crisis_case_id': crisis_case_id,
        'session_id': session.session_id,
        'created_at': firestore.SERVER_TIMESTAMP
    })

    await tool.write_document(f"Company/{session.company_id}/Crises/{crisis_case_id}/Artifacts/{snapshot_id}", snapshot_id, {"artifact_type": "snapshot", "created_at": firestore.SERVER_TIMESTAMP, "payload": snapshot}, session.company_id)
    return snapshot_id

async def write_scorecard_with_session(crisis_case_id: str, scorecard: Dict, session: CrisisSession) -> str:
    """Scorecard creation with dashboard updates"""
    scorecard_id = f"scorecard_{int(time.time())}"
    batch_tool = FirestoreBatchTool()

    operations = [
        WriteOperation(f"Company/{session.company_id}/Crises/{crisis_case_id}/Artifacts/{scorecard_id}", {"artifact_type": "scorecard", "created_at": firestore.SERVER_TIMESTAMP, "payload": scorecard}),
        UpdateOperation(f"companies/{session.company_id}/dashboard/summary", {
            'num_critical': firestore.Increment(1 if scorecard.get('severity', 0) > 0.8 else 0),
            'last_updated': firestore.SERVER_TIMESTAMP
        })
    ]

    await batch_tool.batch_write(operations, session.company_id)
    return scorecard_id

async def update_dashboard_summary_with_session(company_id: str, deltas: Dict[str, int], session: CrisisSession) -> None:
    """Atomic dashboard counter updates with session validation"""
    tool = FirestoreWriteTool()
    await tool.update_counters(f"companies/{company_id}/dashboard/summary", deltas)
```

**Connection Pooling & Performance Optimizations**:

```python
class FirestoreConnectionPool:
    def __init__(self, min_connections=5, max_connections=20):
        self.pool = ConnectionPool(min_connections, max_connections)
        self.query_cache = TTLCache(maxsize=1000, ttl=300)  # 5-minute cache

    async def execute_with_retry(self, operation, max_retries=3):
        # Exponential backoff retry logic
        # Connection health checking
        # Automatic failover support
        pass
```

**Guidelines for FastMap-optimized operations**:

- Batch reads/writes where possible to reduce round trips
- Use connection pooling for concurrent agent operations
- Implement intelligent caching with TTL based on data volatility
- Type-safe operations with schema validation
- Session-scoped security and audit trails
- Optimistic locking for concurrent writes
- Sharded counters for high-volume metrics

## Firestore Security rules (example - allow agents via service account / servers only)

For server-side Admin SDK usage: use IAM roles rather than rules. For client SDKs (dashboard UI) use rules like:

```
service cloud.firestore {
  match /databases/{database}/documents {
    match /companies/{companyId} {
      allow read: if request.auth != null && request.auth.token.company_id == companyId;
      allow write: if false; // only server can write company master data
    }

    match /crises/{crisisId} {
      allow read: if request.auth != null && request.auth.token.company_id == resource.data.company_id;
      allow write: if false; // only server-side agents/admins can write
    }

    // Allow dashboards reads for authorized users
    match /dashboards/{companyId} {
      allow read: if request.auth != null && request.auth.token.company_id == companyId;
      allow write: if false;
    }
  }
}
```

Agents and server processes should use Admin SDK with service account credentials (do not use client credentials).

More: https://firebase.google.com/docs/rules

## Vector DB strategy (Milvus)

- Use Milvus as the single vector store. Choose self-hosted Milvus (K8s or docker-compose) for production scaling or evaluate managed Milvus providers.
- Namespacing & multi-tenancy: either create per-company Milvus collections/partitions or tag vectors with company_id metadata and filter queries. For strict tenant isolation prefer per-company collections.
- Embeddings: precompute embeddings using a chosen embedding model (Vertex Embeddings if available; otherwise open-source or managed embedding models). Upsert vectors with metadata: {case_id, title, date, tags, company_id, source_uri}
- Indexing: pick appropriate index type (HNSW for low-latency/high-recall; IVF+PQ for very large corpora) and tune parameters per collection.
- Retrieval: use top-k vector search with metadata filters. Re-rank candidates in the Recommendation Agent if necessary.
- Ingestion pipeline: provide batch and streaming ingestion with TTL/versioning support. Soft-delete markers and compaction runs help maintain a clean vector corpus.
- Client SDKs: use pymilvus for python ingestion and retrieval. Consider using object storage (GCS) for large payloads and keep only summaries/ids in Milvus metadata.

## Agent communication patterns & ADK-first architecture

ADK (with Vertex AI Agent Engine) provides primitives that make sub-agent communication and orchestration simple and safe. Use ADK workflow agents (SequentialAgent, ParallelAgent, LoopAgent) and `AgentTool` to implement the classifier sub-agents and recommendation pipeline.

Patterns recommended:

1. ADK-native orchestration (preferred): implement the context -> classification -> recommendation pipeline as an ADK `SequentialAgent` or as a `Workflow` composed of LlmAgents and sub-agents. Communication occurs via the shared `session.state`, `InvocationContext.branch`, and ADK Events. This approach benefits from Vertex Agent Engine session management and the VertexAiMemoryBankService for memory.
2. Event-driven augmentation (optional): use Google Pub/Sub when you need durable, cross-service eventing, retries, or integration with systems outside the ADK/Vertex Agent Engine lifecycle (e.g., CI pipelines, external webhook sources, or long-running background workers). Pub/Sub remains useful for decoupling if you expect heavy asynchronous fan-out or long-running jobs that should survive Agent Engine or API restarts.

Design considerations:

- Session & state isolation: create a dedicated ADK `Session` per crisis or per analyst session (session.user_id can be `{company_id}:{crisis_id}`) so that in-memory session.state and Memory Bank searches are scoped. Use `InvocationContext.branch` when running parallel sub-agents to avoid accidental state collisions.
- Memory write discipline: only call `memory_service.add_session_to_memory(session)` after the session completes or after explicit checkpoints to avoid polluting the long-term memory with transient details. Extract and consolidate salient facts before adding them.
- Memory scoping & filters: tag memories with company_id, session_id, and crisis_id. When calling `search_memory`, pass query filters and limit results to the relevant company or session namespace to avoid cross-session leakage.
- Idempotency & concurrency: continue to use Firestore transactions for concurrent writes and ensure agents write to deterministic document IDs when re-running is possible.
- Observability: trace ADK events to a central agent_runs collection and use Vertex/Cloud logging to correlate runs, sessions, and memory writes.

## Observability & monitoring

- Log each agent run with run_id, start_ts, end_ts, status, error
- Metrics: agent_run_count, agent_run_duration_seconds, crisis_severity_histogram
- Store audit logs in crises/{crisis_case_id}/logs and a central agent_runs collection for lifecycle debugging

## Security & data governance

- Use IAM/service accounts for server agents and MCP tool processes
- Never store secrets in Firestore; store them in a secret manager (GCP Secret Manager)
- Limit Firestore read/write rules for client-side SDKs
- Mask PII in logs and embeddings where required (PII handling policy)

## Validation & testing loops

- Unit tests for Firestore CRUD helpers (mock Admin SDK)
- Integration tests using Firestore emulator (Firebase Local Emulator Suite)
- End-to-end functional test that runs a simulation template and asserts recommended plan stored and dashboard updated
- RAG relevance tests for vector retrieval quality (expected top-N similar case IDs)

## Enhanced Success Criteria and Validation Framework

### Functional Requirements Validation

- [ ] **Crisis Simulation Flow**: End-to-end simulation from /simulate POST → complete recommendations ≤ 2 minutes
- [ ] **Context Collection**: All 7 sub-agents execute successfully with >95% data completeness score
- [ ] **Classification Accuracy**: Multi-dimensional scorecard generated with confidence score >0.8
- [ ] **Recommendation Quality**: Strategic plan with ≥3 historical case references and actionable steps
- [ ] **Dashboard Responsiveness**: Real-time updates within 5s of agent completion
- [ ] **Data Consistency**: All agent outputs properly persisted with audit trails

### Performance Requirements Validation

- [ ] **API Response Times**: All endpoints respond within 500ms (excluding agent processing)
- [ ] **Agent Execution**: Context collection ≤30s, Classification ≤45s, Recommendations ≤60s
- [ ] **Database Performance**: Firestore queries optimized with proper indexing (<100ms avg)
- [ ] **Vector Search Performance**: Milvus similarity searches ≤200ms for top-10 results
- [ ] **Concurrent Load**: System handles 10 concurrent crisis simulations without degradation
- [ ] **Memory Efficiency**: Agent memory usage stays within 512MB per session

### Data Quality and Security Validation

- [ ] **Multi-tenancy Isolation**: Company data completely isolated (zero cross-tenant data leakage)
- [ ] **Data Validation**: All documents conform to schema with type safety enforcement
- [ ] **Audit Completeness**: Every agent action logged with execution metadata
- [ ] **Session Management**: ADK sessions properly scoped and cleaned up
- [ ] **Error Handling**: Graceful degradation with proper error reporting and recovery
- [ ] **Access Control**: Service account permissions limited to required operations only

### Integration and Reliability Validation

- [ ] **MCP Tool Integration**: All 15+ MCP tools function correctly with proper error handling
- [ ] **Vector DB Sync**: Milvus metadata synchronized with Firestore consistently
- [ ] **Memory Bank Integration**: Vertex AI Memory Bank properly scoped and searchable
- [ ] **Connection Resilience**: Database connection pooling handles failures and reconnection
- [ ] **Cache Effectiveness**: Intelligent caching reduces database load by ≥40%
- [ ] **Batch Operations**: Batch reads/writes optimize performance for large datasets

### Business Logic Validation

- [ ] **Crisis Classification**: Severity scoring aligns with business rules and thresholds
- [ ] **Stakeholder Mapping**: Influence scores and communication preferences accurate
- [ ] **Historical Context**: Pattern recognition identifies relevant precedents correctly
- [ ] **Recommendation Relevance**: Strategic plans are actionable and context-appropriate
- [ ] **Resource Estimation**: Cost and timeline estimates within realistic ranges
- [ ] **Compliance Validation**: All recommendations pass legal and regulatory checks

## Deliverables for implementation

- FastAPI scaffold with endpoints listed above
- Firestore CRUD helper library (python) with tests
- ADK agent specs (tool contracts) and example implementations using Vertex AI + ADK primitives (LLM agents, AgentTool, SequentialAgent) and memory-service wiring
- Milvus ingestion pipeline (ingest case studies + embeddings) and sample local dev setup (docker-compose or docker-based Milvus)
- CI tests using Firestore emulator and a small Milvus test instance (docker) for retrieval smoke tests
- Example simulation templates and history entries

## Next steps / Deployment notes

1. Create Firestore project and service account; provision roles for ADK/Agent Engine service accounts and Vertex AI permissions
2. Provision Milvus for dev (docker-compose) and plan production hosting (self-hosted K8s cluster or managed Milvus)
3. Implement FastAPI scaffold and CRUD helpers; run with Firestore emulator during development
4. Implement ADK agents locally and wire `VertexAiMemoryBankService` for long-term memory. Deploy ADK app to Vertex AI Agent Engine when ready.
5. Use Pub/Sub only when durable cross-system eventing, external triggers, or long-lived background processing is required; otherwise prefer ADK-native session orchestration

## References (authoritative)

- FastAPI docs: https://fastapi.tiangolo.com/
- Firestore docs: https://cloud.google.com/firestore/docs/overview
- Firebase Admin SDK: https://firebase.google.com/docs/admin/setup
- Firestore Security Rules: https://firebase.google.com/docs/rules
- Vertex AI Agent Engine / ADK deploy: https://google.github.io/adk-docs/deploy/agent-engine/
- ADK Memory (Vertex AI Memory Bank): https://google.github.io/adk-docs/sessions/memory/
- ADK multi-agent / workflow patterns: https://google.github.io/adk-docs/agents/multi-agents/
- Milvus docs: https://milvus.io/docs/overview.md
- Milvus python client / install: https://milvus.io/docs/install_standalone-docker.md

---

## Change log

- 2025-08-15: initial PRP created by template generator
