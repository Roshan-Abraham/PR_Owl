"""
Context Collector Agent (Agent A) with 7 Specialized Sub-agents
Implements comprehensive context collection for crisis cases using ADK orchestration
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import structlog

# ADK imports with proper mock implementations
from agents.adk_mocks import (
    ADK_AVAILABLE, BaseAgent, SequentialAgent, LlmAgent, AgentTool, 
    Session, InvocationContext, VertexAiMemoryBankService
)

from models.schemas import CrisisSnapshot, CrisisContext, CompanyProfile, CompanyEvent
from tools.mcp_tools import (
    FirestoreReadTool, CompanyProfileValidator, StakeholderAnalyzer, TimelineBuilder
)
from tools.vector_search_tool import VectorSearchTool
from infrastructure.firestore_client import FirestoreConnectionPool

logger = structlog.get_logger()


class ContextCollectorAgent:
    """
    Main orchestrator for context collection with 7 specialized sub-agents
    Uses ADK SequentialAgent to coordinate sub-agent execution
    """

    def __init__(self, db_pool: FirestoreConnectionPool, vector_search: VectorSearchTool) -> None:
        self.db_pool = db_pool
        self.vector_search = vector_search

        # Initialize MCP tools
        self.firestore_read_tool = FirestoreReadTool(db_pool)
        self.profile_validator = CompanyProfileValidator()
        self.stakeholder_analyzer = StakeholderAnalyzer()
        self.timeline_builder = TimelineBuilder()

        # Performance metrics
        self.execution_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "avg_execution_time_ms": 0.0
        }

        # Initialize sub-agents
        self.sub_agents = self._initialize_sub_agents()

    def _initialize_sub_agents(self) -> List[Any]:
        """Initialize the 7 specialized sub-agents"""
        return [
            CompanyProfileAgent(self.firestore_read_tool,
                                self.profile_validator),
            StakeholderMappingAgent(
                self.firestore_read_tool, self.stakeholder_analyzer),
            EventContextAgent(self.firestore_read_tool, self.timeline_builder),
            HistoricalPatternAgent(self.firestore_read_tool),
            ExternalSignalsAgent(),
            KnowledgeBaseAgent(self.firestore_read_tool, self.vector_search),
            SnapshotSynthesizerAgent(self.db_pool)
        ]

    async def collect_context(self, crisis_case_id: str, company_id: str, session_id: str) -> str:
        """
        Main entry point for context collection
        Orchestrates all sub-agents and returns snapshot_id
        """
        start_time = time.time()
        self.execution_metrics["total_executions"] += 1
        
        # Create trace ID for sub-agent orchestration
        trace_id = f"context_collection_{crisis_case_id}_{int(start_time)}"
        
        from infrastructure.monitoring import structured_logger
        structured_logger.start_trace(
            trace_id, "context_collection_orchestration",
            crisis_case_id=crisis_case_id, company_id=company_id, session_id=session_id
        )

        try:
            logger.info("Starting context collection",
                        crisis_case_id=crisis_case_id, company_id=company_id, trace_id=trace_id)

            # Create ADK session for context
            session = Session(session_id)
            session.state = {
                "crisis_case_id": crisis_case_id,
                "company_id": company_id,
                "context_data": {},
                "execution_start": datetime.utcnow().isoformat()
            }

            # Execute sub-agents sequentially
            context_data = await self._execute_sub_agents(session, trace_id)

            # Create final snapshot
            snapshot_id = await self._create_snapshot(crisis_case_id, context_data, session_id)

            # Update execution metrics
            execution_time = (time.time() - start_time) * 1000
            self.execution_metrics["successful_executions"] += 1
            self.execution_metrics["avg_execution_time_ms"] = (
                (self.execution_metrics["avg_execution_time_ms"] * (self.execution_metrics["total_executions"] - 1)
                 + execution_time) / self.execution_metrics["total_executions"]
            )
            
            # End trace successfully
            trace_summary = structured_logger.end_trace(
                trace_id, "success",
                total_execution_time_ms=execution_time,
                sub_agents_executed=len(self.sub_agents),
                snapshot_id=snapshot_id
            )

            logger.info("Context collection completed",
                        crisis_case_id=crisis_case_id, snapshot_id=snapshot_id,
                        trace_id=trace_id, execution_time_ms=execution_time)

            return snapshot_id

        except Exception as e:
            self.execution_metrics["failed_executions"] += 1
            execution_time = (time.time() - start_time) * 1000
            
            # End trace with failure
            structured_logger.end_trace(
                trace_id, "failed",
                total_execution_time_ms=execution_time,
                error_details=str(e),
                error_type=type(e).__name__
            )
            
            logger.error("Context collection failed",
                         crisis_case_id=crisis_case_id, trace_id=trace_id, error=str(e))
            raise

    async def _execute_sub_agents(self, session: Session, trace_id: str) -> Dict[str, Any]:
        """Execute all sub-agents and collect their outputs with structured logging"""
        context_data = {}

        for i, sub_agent in enumerate(self.sub_agents):
            sub_agent_start = time.time()
            agent_name = sub_agent.__class__.__name__.lower().replace('agent', '')
            
            try:
                logger.debug(
                    f"Executing sub-agent {i+1}/7: {sub_agent.__class__.__name__}",
                    trace_id=trace_id)

                # Execute sub-agent
                sub_agent_result = await sub_agent.execute(session)
                sub_agent_duration = time.time() - sub_agent_start
                
                # Log sub-agent execution
                structured_logger.log_sub_agent_execution(
                    trace_id, "context_collector", agent_name, i+1, "success",
                    sub_agent_duration, session.state.get("crisis_case_id"), 
                    session.state.get("company_id"),
                    result_summary=f"Sub-agent completed with {len(str(sub_agent_result))} characters of data"
                )

                # Store result in session state
                context_data[agent_name] = sub_agent_result
                session.state["context_data"][agent_name] = sub_agent_result

                logger.debug(
                    f"Sub-agent {i+1}/7 completed: {sub_agent.__class__.__name__}",
                    trace_id=trace_id, duration_seconds=sub_agent_duration)

            except Exception as e:
                sub_agent_duration = time.time() - sub_agent_start
                error_msg = str(e)
                
                # Log sub-agent failure
                structured_logger.log_sub_agent_execution(
                    trace_id, "context_collector", agent_name, i+1, "failed",
                    sub_agent_duration, session.state.get("crisis_case_id"), 
                    session.state.get("company_id"),
                    error_details=error_msg
                )
                
                logger.error(f"Sub-agent {i+1}/7 failed: {sub_agent.__class__.__name__}",
                             error=error_msg, trace_id=trace_id, duration_seconds=sub_agent_duration)
                             
                # Continue with other sub-agents, but log the failure
                context_data[f"{agent_name}_error"] = error_msg

        return context_data

    async def _create_snapshot(self, crisis_case_id: str, context_data: Dict[str, Any],
                               session_id: str) -> str:
        """Create the final crisis snapshot document"""

        # Build structured context
        crisis_context = CrisisContext(
            company_profile=context_data.get('companyprofile', {}),
            recent_events=context_data.get(
                'eventcontext', {}).get('timeline', []),
            relations=context_data.get(
                'stakeholdermapping', {}).get('stakeholders', []),
            social_context=context_data.get('externalsignals', {}),
            last_24h_activity=context_data.get(
                'eventcontext', {}).get('recent_activity', {})
        )

        # Create snapshot
        snapshot = CrisisSnapshot(
            crisis_case_id=crisis_case_id,
            context=crisis_context,
            agent_session_id=session_id,
            processing_metadata={
                "sub_agents_executed": list(context_data.keys()),
                "execution_timestamp": datetime.utcnow().isoformat(),
                "data_completeness_score": self._calculate_completeness_score(context_data)
            }
        )

        # Store in Firestore under the company's crisis Artifacts subcollection
        snapshot_path = f"Company/{session.state.get('company_id')}/Crises/{crisis_case_id}/Artifacts/{snapshot.snapshot_id}"
        await self.db_pool.set_document(snapshot_path, snapshot.model_dump())

        # Update crisis case with latest snapshot reference (company-scoped path)
        await self.db_pool.update_document(
            f"Company/{session.state.get('company_id')}/Crises/{crisis_case_id}",
            {
                "snapshot_id": snapshot.snapshot_id,
                "current_status": "context_collected",
                "updated_at": datetime.utcnow()
            },
            company_id=session.state.get('company_id')
        )

        return snapshot.snapshot_id

    def _calculate_completeness_score(self, context_data: Dict[str, Any]) -> float:
        """Calculate data completeness score based on sub-agent results"""
        total_agents = len(self.sub_agents)
        successful_agents = len(
            [k for k in context_data.keys() if not k.endswith('_error')])

        # Base score from successful agent execution
        base_score = successful_agents / total_agents

        # Bonus points for data quality
        quality_bonus = 0.0

        if 'companyprofile' in context_data:
            profile_score = context_data['companyprofile'].get(
                'completeness_score', 0)
            quality_bonus += profile_score * 0.1

        if 'stakeholdermapping' in context_data:
            stakeholder_count = len(
                context_data['stakeholdermapping'].get('stakeholders', []))
            quality_bonus += min(0.1, stakeholder_count / 10)

        return min(1.0, base_score + quality_bonus)

# Specialized Sub-agents


class CompanyProfileAgent:
    """Company Core Data Specialist - Agent 1/7"""

    def __init__(self, firestore_read_tool: FirestoreReadTool, validator: CompanyProfileValidator) -> None:
        self.read_tool = firestore_read_tool
        self.validator = validator

    async def execute(self, session: Session) -> Dict[str, Any]:
        """Extract and validate company profile data"""
        company_id = session.state.get("company_id")

        try:
            # Read company profile
            company_data = await self.read_tool.read_document(
                "companies", company_id, company_id
            )

            if not company_data:
                raise ValueError(f"Company profile not found: {company_id}")

            # Read company details
            details_data = await self.read_tool.read_document(
                f"companies/{company_id}/details", "profile", company_id
            )

            # Validate profile completeness
            validation_result = await self.validator.validate_profile(company_data)

            return {
                "company_profile": company_data,
                "company_details": details_data or {},
                "validation": validation_result,
                "completeness_score": validation_result.get("completeness_score", 0.0)
            }

        except Exception as e:
            logger.error("CompanyProfileAgent failed",
                         company_id=company_id, error=str(e))
            return {"error": str(e), "completeness_score": 0.0}


class StakeholderMappingAgent:
    """Relationship Intelligence Specialist - Agent 2/7"""

    def __init__(self, firestore_read_tool: FirestoreReadTool, analyzer: StakeholderAnalyzer) -> None:
        self.read_tool = firestore_read_tool
        self.analyzer = analyzer

    async def execute(self, session: Session) -> Dict[str, Any]:
        """Map stakeholder relationships and analyze influence"""
        company_id = session.state.get("company_id")

        try:
            # Query stakeholder relations
            relations = await self.read_tool.query_collection(
                f"companies/{company_id}/relations",
                filters=[],
                company_id=company_id
            )

            # Analyze influence network
            influence_analysis = await self.analyzer.analyze_influence_network(relations)

            return {
                "stakeholders": relations,
                "influence_analysis": influence_analysis,
                "total_stakeholders": len(relations),
                "key_influencers": influence_analysis.get("key_influencers", [])
            }

        except Exception as e:
            logger.error("StakeholderMappingAgent failed",
                         company_id=company_id, error=str(e))
            return {"error": str(e), "stakeholders": []}


class EventContextAgent:
    """Temporal Context Specialist - Agent 3/7"""

    def __init__(self, firestore_read_tool: FirestoreReadTool, timeline_builder: TimelineBuilder) -> None:
        self.read_tool = firestore_read_tool
        self.timeline_builder = timeline_builder

    async def execute(self, session: Session) -> Dict[str, Any]:
        """Analyze recent events and build timeline"""
        company_id = session.state.get("company_id")

        try:
            # Query recent events (last 30 days)
            cutoff_date = datetime.utcnow() - timedelta(days=30)

            events = await self.read_tool.query_collection(
                f"companies/{company_id}/events",
                filters=[("start_time", ">=", cutoff_date)],
                company_id=company_id,
                options={"order_by": "start_time",
                         "order_direction": "desc", "limit": 50}
            )

            # Build timeline and analyze patterns
            timeline_analysis = await self.timeline_builder.build_event_timeline(events)

            # Extract last 24h activity
            last_24h = datetime.utcnow() - timedelta(hours=24)
            recent_activity = [
                e for e in events
                if e.get('start_time', datetime.min) >= last_24h
            ]

            return {
                "timeline": timeline_analysis.get("timeline", []),
                "recent_events": events[:10],  # Most recent 10
                "recent_activity": {
                    "last_24h_events": len(recent_activity),
                    "events": recent_activity
                },
                "patterns": timeline_analysis.get("patterns", {}),
                "analysis_period_days": 30
            }

        except Exception as e:
            logger.error("EventContextAgent failed",
                         company_id=company_id, error=str(e))
            return {"error": str(e), "timeline": [], "recent_events": []}


class HistoricalPatternAgent:
    """Crisis History Specialist - Agent 4/7"""

    def __init__(self, firestore_read_tool: FirestoreReadTool) -> None:
        self.read_tool = firestore_read_tool

    async def execute(self, session: Session) -> Dict[str, Any]:
        """Analyze historical crisis patterns"""
        company_id = session.state.get("company_id")

        try:
            # Query historical crises (last 12 months)
            cutoff_date = datetime.utcnow() - timedelta(days=365)

            historical_crises = await self.read_tool.query_collection(
                "crises",
                filters=[
                    ("company_id", "==", company_id),
                    ("created_at", ">=", cutoff_date),
                    ("current_status", "in", ["resolved", "archived"])
                ],
                company_id=company_id,
                options={"order_by": "severity_score",
                         "order_direction": "desc", "limit": 20}
            )

            # Analyze patterns
            patterns = self._analyze_crisis_patterns(historical_crises)

            return {
                "historical_crises": historical_crises,
                "patterns": patterns,
                "total_historical_crises": len(historical_crises),
                "analysis_period_days": 365
            }

        except Exception as e:
            logger.error("HistoricalPatternAgent failed",
                         company_id=company_id, error=str(e))
            return {"error": str(e), "historical_crises": []}

    def _analyze_crisis_patterns(self, crises: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in historical crises"""
        if not crises:
            return {}

        # Analyze by nature
        by_nature = {}
        severity_scores = []
        resolution_times = []

        for crisis in crises:
            nature = crisis.get("nature", "unknown")
            by_nature[nature] = by_nature.get(nature, 0) + 1

            if severity := crisis.get("severity_score"):
                severity_scores.append(severity)

            if res_time := crisis.get("estimated_resolution_time_hours"):
                resolution_times.append(res_time)

        return {
            "crisis_types": by_nature,
            "average_severity": sum(severity_scores) / len(severity_scores) if severity_scores else 0,
            "average_resolution_hours": sum(resolution_times) / len(resolution_times) if resolution_times else 0,
            "most_common_type": max(by_nature.items(), key=lambda x: x[1])[0] if by_nature else None
        }


class ExternalSignalsAgent:
    """External Intelligence Specialist - Agent 5/7"""

    async def execute(self, session: Session) -> Dict[str, Any]:
        """Collect external signals and market intelligence"""
        company_id = session.state.get("company_id")

        try:
            # Mock external signals collection
            # In production, this would integrate with:
            # - News APIs (NewsAPI, Google News)
            # - Social media monitoring (Twitter API, Reddit)
            # - Industry reports and feeds
            # - Market data services

            external_signals = {
                "news_sentiment": {
                    "overall_score": 0.2,  # Slightly negative
                    "recent_mentions": 15,
                    "trending_topics": ["product_issue", "customer_complaints"]
                },
                "social_media": {
                    "mention_volume": 230,
                    "sentiment_distribution": {"positive": 0.3, "neutral": 0.4, "negative": 0.3},
                    "key_hashtags": ["#productrecall", "#customerservice"]
                },
                "industry_context": {
                    "sector_health": 0.7,
                    "competitive_activity": ["competitor_launch", "price_changes"],
                    "regulatory_developments": []
                },
                "market_indicators": {
                    "stock_movement": -0.02,  # 2% down
                    "volume_spike": True,
                    "analyst_sentiment": "hold"
                }
            }

            return {
                "external_signals": external_signals,
                "signal_strength": self._calculate_signal_strength(external_signals),
                "last_updated": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error("ExternalSignalsAgent failed",
                         company_id=company_id, error=str(e))
            return {"error": str(e), "external_signals": {}}

    def _calculate_signal_strength(self, signals: Dict[str, Any]) -> float:
        """Calculate overall external signal strength"""
        news_weight = 0.3
        social_weight = 0.3
        market_weight = 0.4

        news_score = abs(signals.get(
            "news_sentiment", {}).get("overall_score", 0))
        social_score = signals.get("social_media", {}).get(
            "mention_volume", 0) / 1000  # Normalize
        market_score = abs(signals.get("market_indicators",
                           {}).get("stock_movement", 0)) * 10

        return min(1.0, news_score * news_weight + social_score * social_weight + market_score * market_weight)


class KnowledgeBaseAgent:
    """Internal Knowledge Specialist - Agent 6/7"""

    def __init__(self, firestore_read_tool: FirestoreReadTool, vector_search: VectorSearchTool) -> None:
        self.read_tool = firestore_read_tool
        self.vector_search = vector_search

    async def execute(self, session: Session) -> Dict[str, Any]:
        """Search company knowledge base for relevant information"""
        company_id = session.state.get("company_id")
        crisis_case_id = session.state.get("crisis_case_id")

        try:
            # Query company knowledge base
            kb_entries = await self.read_tool.query_collection(
                f"companies/{company_id}/knowledge_base",
                filters=[],
                company_id=company_id,
                options={"limit": 10}
            )

            # Vector search for relevant case studies
            if crisis_case_id:
                search_query = f"crisis case {crisis_case_id} similar situations lessons learned"
                similar_cases = await self.vector_search.similarity_search(
                    search_query,
                    filters={"company_scoped": True, "source_types": [
                        "case_study", "internal_knowledge"]},
                    company_id=company_id,
                    top_k=5
                )
            else:
                similar_cases = []

            return {
                "knowledge_base_entries": kb_entries,
                "similar_cases": [asdict(case) for case in similar_cases],
                "total_kb_entries": len(kb_entries),
                "relevant_cases_found": len(similar_cases)
            }

        except Exception as e:
            logger.error("KnowledgeBaseAgent failed",
                         company_id=company_id, error=str(e))
            return {"error": str(e), "knowledge_base_entries": [], "similar_cases": []}


class SnapshotSynthesizerAgent:
    """Data Integration Specialist - Agent 7/7"""

    def __init__(self, db_pool: FirestoreConnectionPool) -> None:
        self.db_pool = db_pool

    async def execute(self, session: Session) -> Dict[str, Any]:
        """Synthesize and validate collected context data"""
        context_data = session.state.get("context_data", {})

        try:
            # Data quality validation
            quality_report = self._validate_data_quality(context_data)

            # Data synthesis
            synthesized_insights = self._synthesize_insights(context_data)

            # Generate data lineage
            lineage = self._generate_data_lineage(context_data)

            return {
                "synthesis_report": {
                    "data_quality": quality_report,
                    "synthesized_insights": synthesized_insights,
                    "data_lineage": lineage,
                    "synthesis_timestamp": datetime.utcnow().isoformat()
                }
            }

        except Exception as e:
            logger.error("SnapshotSynthesizerAgent failed", error=str(e))
            return {"error": str(e)}

    def _validate_data_quality(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the quality of collected context data"""
        quality_scores = {}

        for agent_name, data in context_data.items():
            if isinstance(data, dict) and "error" not in data:
                # Simple quality scoring based on data presence
                if agent_name == "companyprofile":
                    quality_scores[agent_name] = data.get(
                        "completeness_score", 0.0)
                elif agent_name == "stakeholdermapping":
                    quality_scores[agent_name] = min(
                        1.0, len(data.get("stakeholders", [])) / 5)
                elif agent_name == "eventcontext":
                    quality_scores[agent_name] = min(
                        1.0, len(data.get("recent_events", [])) / 3)
                else:
                    quality_scores[agent_name] = 0.8 if data else 0.0
            else:
                quality_scores[agent_name] = 0.0

        overall_quality = sum(quality_scores.values()) / \
            len(quality_scores) if quality_scores else 0.0

        return {
            "individual_scores": quality_scores,
            "overall_quality_score": overall_quality,
            "quality_threshold_met": overall_quality >= 0.7
        }

    def _synthesize_insights(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level insights from collected data"""
        insights = {}

        # Company readiness
        if "companyprofile" in context_data:
            profile_data = context_data["companyprofile"]
            insights["company_readiness"] = {
                "profile_complete": profile_data.get("completeness_score", 0) > 0.7,
                "stakeholder_network_size": len(context_data.get("stakeholdermapping", {}).get("stakeholders", [])),
                "recent_activity_level": len(context_data.get("eventcontext", {}).get("recent_events", []))
            }

        # Risk indicators
        if "externalsignals" in context_data:
            signals = context_data["externalsignals"].get(
                "external_signals", {})
            insights["risk_indicators"] = {
                "external_pressure": signals.get("news_sentiment", {}).get("overall_score", 0) < -0.3,
                "social_media_attention": signals.get("social_media", {}).get("mention_volume", 0) > 500,
                "market_impact": abs(signals.get("market_indicators", {}).get("stock_movement", 0)) > 0.05
            }

        return insights

    def _generate_data_lineage(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data lineage information"""
        return {
            "sources": list(context_data.keys()),
            "collection_timestamp": datetime.utcnow().isoformat(),
            "data_freshness": {
                agent: "current" if "error" not in str(data) else "error"
                for agent, data in context_data.items()
            }
        }


# Export
__all__ = ['ContextCollectorAgent']
