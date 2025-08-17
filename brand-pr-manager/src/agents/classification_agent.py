"""
Classification Agent (Agent B) with 7 Multi-Dimensional Analysis Sub-agents
Implements comprehensive crisis classification using ADK ParallelAgent and SequentialAgent orchestration
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import structlog

# ADK imports (placeholder - would use actual ADK in production)
try:
    from google.adk import ParallelAgent, SequentialAgent, LlmAgent, AgentTool, Session, InvocationContext
    from google.adk.memory import VertexAiMemoryBankService
except ImportError:
    # Mock ADK classes for development
    class ParallelAgent:
        def __init__(self, agents: List[Any]):
            self.agents = agents

    class SequentialAgent:
        def __init__(self, agents: List[Any]):
            self.agents = agents

    class LlmAgent:
        def __init__(self, name: str, prompt: str, tools: List[Any] = None):
            self.name = name
            self.prompt = prompt
            self.tools = tools or []

    class AgentTool:
        def __init__(self, name: str = "", description: str = ""):
            self.name = name
            self.description = description

    class Session:
        def __init__(self, session_id: str):
            self.session_id = session_id
            self.state = {}

    class InvocationContext:
        def __init__(self, session: Session):
            self.session = session

from models.schemas import Scorecard, ScorecardMetrics, AffectedEntity, CrisisSnapshot
from tools.mcp_tools import FirestoreReadTool, FirestoreWriteTool
from infrastructure.firestore_client import FirestoreConnectionPool

logger = structlog.get_logger()


class ClassificationAgent:
    """
    Main orchestrator for crisis classification with 7 specialized analysis sub-agents
    Uses ADK ParallelAgent for concurrent analysis + SequentialAgent for synthesis
    """

    def __init__(self, db_pool: FirestoreConnectionPool) -> None:
        self.db_pool = db_pool

        # Initialize MCP tools
        self.firestore_read_tool = FirestoreReadTool(db_pool)
        self.firestore_write_tool = FirestoreWriteTool(db_pool)

        # Performance metrics
        self.execution_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "avg_execution_time_ms": 0.0
        }

        # Initialize sub-agents
        self.analysis_agents = self._initialize_analysis_agents()
        self.synthesizer_agent = ScorecardSynthesizerAgent(db_pool)

    def _initialize_analysis_agents(self) -> List[Any]:
        """Initialize the 7 specialized analysis sub-agents"""
        return [
            SeverityAssessmentAgent(self.firestore_read_tool),
            ImpactPredictionAgent(self.firestore_read_tool),
            StakeholderExposureAgent(self.firestore_read_tool),
            TimelineAnalysisAgent(self.firestore_read_tool),
            CompetitiveContextAgent(self.firestore_read_tool),
            LegalComplianceAgent(self.firestore_read_tool),
            RiskIntegrationAgent(self.firestore_read_tool)
        ]

    async def classify_crisis(self, snapshot_id: str, crisis_case_id: str,
                              company_id: str, session_id: str) -> str:
        """
        Main entry point for crisis classification
        Returns scorecard_id after complete multi-dimensional analysis
        """
        start_time = time.time()
        self.execution_metrics["total_executions"] += 1

        try:
            logger.info("Starting crisis classification",
                        crisis_case_id=crisis_case_id, snapshot_id=snapshot_id)

            # Create ADK session for classification
            session = Session(session_id)
            session.state = {
                "crisis_case_id": crisis_case_id,
                "company_id": company_id,
                "snapshot_id": snapshot_id,
                "classification_start": datetime.utcnow().isoformat(),
                "analysis_results": {}
            }

            # Load crisis snapshot for analysis
            snapshot_data = await self._load_snapshot(crisis_case_id, snapshot_id, company_id)
            session.state["snapshot_data"] = snapshot_data

            # Execute parallel analysis sub-agents
            analysis_results = await self._execute_parallel_analysis(session)

            # Synthesize results into final scorecard
            scorecard_id = await self._synthesize_scorecard(session, analysis_results)

            # Update execution metrics
            execution_time = (time.time() - start_time) * 1000
            self.execution_metrics["successful_executions"] += 1
            self.execution_metrics["avg_execution_time_ms"] = (
                (self.execution_metrics["avg_execution_time_ms"] * (self.execution_metrics["total_executions"] - 1)
                 + execution_time) / self.execution_metrics["total_executions"]
            )

            logger.info("Crisis classification completed",
                        crisis_case_id=crisis_case_id, scorecard_id=scorecard_id,
                        execution_time_ms=execution_time)

            return scorecard_id

        except Exception as e:
            self.execution_metrics["failed_executions"] += 1
            logger.error("Crisis classification failed",
                         crisis_case_id=crisis_case_id, error=str(e))
            raise

    async def _load_snapshot(self, crisis_case_id: str, snapshot_id: str, company_id: str) -> Dict[str, Any]:
        """Load the crisis snapshot for analysis"""
        # Read snapshot from company-scoped Artifacts subcollection
        snapshot_path = f"Company/{company_id}/Crises/{crisis_case_id}/Artifacts/{snapshot_id}"
        snapshot_data = await self.firestore_read_tool.read_document(
            f"Company/{company_id}/Crises/{crisis_case_id}/Artifacts", snapshot_id, company_id
        )

        if not snapshot_data:
            raise ValueError(f"Crisis snapshot not found: {snapshot_id}")

        return snapshot_data

    async def _execute_parallel_analysis(self, session: Session) -> Dict[str, Any]:
        """Execute all analysis sub-agents in parallel for efficiency"""
        analysis_tasks = []

        # Create parallel execution tasks
        for agent in self.analysis_agents:
            task = asyncio.create_task(
                self._safe_agent_execution(agent, session),
                name=f"analysis_{agent.__class__.__name__}"
            )
            analysis_tasks.append(task)

        # Wait for all analyses to complete
        analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

        # Process results and handle exceptions
        processed_results = {}
        for i, (agent, result) in enumerate(zip(self.analysis_agents, analysis_results)):
            agent_name = agent.__class__.__name__.lower().replace('agent', '')

            if isinstance(result, Exception):
                logger.error(f"Analysis agent {i+1}/7 failed: {agent.__class__.__name__}",
                             error=str(result))
                processed_results[agent_name] = {
                    "error": str(result), "status": "failed"}
            else:
                processed_results[agent_name] = {
                    "data": result, "status": "success"}
                logger.debug(
                    f"Analysis agent {i+1}/7 completed: {agent.__class__.__name__}")

        return processed_results

    async def _safe_agent_execution(self, agent: Any, session: Session) -> Dict[str, Any]:
        """Safely execute an agent with error handling"""
        try:
            return await agent.execute(session)
        except Exception as e:
            logger.warning(
                f"Agent execution failed: {agent.__class__.__name__}", error=str(e))
            raise

    async def _synthesize_scorecard(self, session: Session, analysis_results: Dict[str, Any]) -> str:
        """Synthesize analysis results into final scorecard"""
        session.state["analysis_results"] = analysis_results

        # Execute synthesizer agent
        scorecard_data = await self.synthesizer_agent.execute(session)

        return scorecard_data.get("scorecard_id")

# Specialized Analysis Sub-agents


class SeverityAssessmentAgent:
    """Crisis Magnitude Specialist - Agent 1/7"""

    def __init__(self, firestore_read_tool: FirestoreReadTool) -> None:
        self.read_tool = firestore_read_tool

    async def execute(self, session: Session) -> Dict[str, Any]:
        """Assess crisis magnitude with multi-dimensional severity scoring"""
        snapshot_data = session.state.get("snapshot_data", {})
        company_id = session.state.get("company_id")

        try:
            context = snapshot_data.get("context", {})

            # Analyze multiple severity dimensions
            scale_score = self._assess_scale(context)
            urgency_score = self._assess_urgency(context)
            complexity_score = self._assess_complexity(context)
            cascade_potential = self._assess_cascade_potential(context)

            # Calculate overall severity
            severity_weights = {"scale": 0.3, "urgency": 0.3,
                                "complexity": 0.2, "cascade": 0.2}
            overall_severity = (
                scale_score * severity_weights["scale"] +
                urgency_score * severity_weights["urgency"] +
                complexity_score * severity_weights["complexity"] +
                cascade_potential * severity_weights["cascade"]
            )

            return {
                "severity_analysis": {
                    "scale": scale_score,
                    "urgency": urgency_score,
                    "complexity": complexity_score,
                    "cascade_potential": cascade_potential,
                    "overall_severity": overall_severity,
                    "confidence_interval": {"lower": max(0, overall_severity - 0.15),
                                            "upper": min(1, overall_severity + 0.15)}
                },
                "severity_factors": self._identify_severity_factors(context),
                "benchmark_comparison": await self._benchmark_against_historical(company_id, overall_severity)
            }

        except Exception as e:
            logger.error("SeverityAssessmentAgent failed", error=str(e))
            return {"error": str(e), "severity_analysis": {"overall_severity": 0.5}}

    def _assess_scale(self, context: Dict[str, Any]) -> float:
        """Assess the scale/scope of the crisis"""
        stakeholders = context.get("relations", [])
        events = context.get("recent_events", [])

        # Base scale on stakeholder count and event frequency
        stakeholder_impact = min(
            1.0, len(stakeholders) / 10)  # Normalize to 0-1
        # Recent events indicate scale
        event_frequency = min(1.0, len(events) / 5)

        # Check external signals
        external_signals = context.get(
            "social_context", {}).get("external_signals", {})
        media_attention = external_signals.get(
            "social_media", {}).get("mention_volume", 0)
        media_impact = min(1.0, media_attention / 1000)

        return (stakeholder_impact * 0.4 + event_frequency * 0.3 + media_impact * 0.3)

    def _assess_urgency(self, context: Dict[str, Any]) -> float:
        """Assess time-sensitive aspects of the crisis"""
        last_24h = context.get("last_24h_activity", {})
        event_count = last_24h.get("last_24h_events", 0)

        # More recent activity = higher urgency
        urgency_score = min(1.0, event_count / 3)

        # Check market indicators for urgency signals
        external_signals = context.get(
            "social_context", {}).get("external_signals", {})
        market_movement = abs(external_signals.get(
            "market_indicators", {}).get("stock_movement", 0))
        # Stock movement indicates urgency
        market_urgency = min(1.0, market_movement * 10)

        return max(urgency_score, market_urgency)

    def _assess_complexity(self, context: Dict[str, Any]) -> float:
        """Assess the complexity of managing this crisis"""
        stakeholders = context.get("relations", [])
        stakeholder_types = set(s.get("type", "unknown") for s in stakeholders)

        # More stakeholder types = higher complexity
        type_complexity = min(1.0, len(stakeholder_types) / 5)

        # Company profile complexity
        company_profile = context.get("company_profile", {})
        industry_complexity = 0.7 if company_profile.get(
            "industry") in ["finance", "healthcare", "government"] else 0.3

        return (type_complexity * 0.6 + industry_complexity * 0.4)

    def _assess_cascade_potential(self, context: Dict[str, Any]) -> float:
        """Assess potential for crisis to cascade/escalate"""
        # High-influence stakeholders increase cascade potential
        stakeholders = context.get("relations", [])
        high_influence_count = len(
            [s for s in stakeholders if s.get("importance_score", 0) > 0.8])
        influence_cascade = min(1.0, high_influence_count / 3)

        # Industry factors
        company_profile = context.get("company_profile", {})
        high_cascade_industries = ["finance",
                                   "healthcare", "media", "government"]
        industry_cascade = 0.8 if company_profile.get(
            "industry") in high_cascade_industries else 0.4

        return (influence_cascade * 0.6 + industry_cascade * 0.4)

    def _identify_severity_factors(self, context: Dict[str, Any]) -> List[str]:
        """Identify key factors contributing to severity"""
        factors = []

        stakeholders = context.get("relations", [])
        if len([s for s in stakeholders if s.get("importance_score", 0) > 0.8]) > 2:
            factors.append("Multiple high-influence stakeholders involved")

        last_24h = context.get("last_24h_activity", {})
        if last_24h.get("last_24h_events", 0) > 2:
            factors.append("High recent activity volume")

        external_signals = context.get(
            "social_context", {}).get("external_signals", {})
        if external_signals.get("social_media", {}).get("mention_volume", 0) > 500:
            factors.append("Significant social media attention")

        return factors

    async def _benchmark_against_historical(self, company_id: str, current_severity: float) -> Dict[str, Any]:
        """Compare current severity against historical patterns"""
        try:
            # Query historical crises for benchmarking
            historical_crises = await self.read_tool.query_collection(
                "crises",
                filters=[("company_id", "==", company_id),
                         ("current_status", "in", ["resolved", "archived"])],
                company_id=company_id,
                options={"limit": 20}
            )

            if not historical_crises:
                return {"benchmark": "no_historical_data"}

            severity_scores = [c.get("severity_score", 0)
                               for c in historical_crises]
            avg_severity = sum(severity_scores) / len(severity_scores)

            percentile_rank = len(
                [s for s in severity_scores if s < current_severity]) / len(severity_scores)

            return {
                "historical_average": avg_severity,
                "percentile_rank": percentile_rank,
                "severity_category": "high" if percentile_rank > 0.75 else "medium" if percentile_rank > 0.25 else "low"
            }

        except Exception as e:
            logger.warning("Historical benchmarking failed", error=str(e))
            return {"benchmark": "comparison_failed"}


class ImpactPredictionAgent:
    """Consequence Analysis Specialist - Agent 2/7"""

    def __init__(self, firestore_read_tool: FirestoreReadTool) -> None:
        self.read_tool = firestore_read_tool

    async def execute(self, session: Session) -> Dict[str, Any]:
        """Predict multi-dimensional impact with probability distributions"""
        snapshot_data = session.state.get("snapshot_data", {})

        try:
            context = snapshot_data.get("context", {})

            # Analyze impact across dimensions
            financial_impact = self._predict_financial_impact(context)
            reputational_impact = self._predict_reputational_impact(context)
            operational_impact = self._predict_operational_impact(context)
            regulatory_impact = self._predict_regulatory_impact(context)
            market_impact = self._predict_market_impact(context)

            # Generate impact scenarios
            scenarios = self._generate_impact_scenarios([
                financial_impact, reputational_impact, operational_impact,
                regulatory_impact, market_impact
            ])

            return {
                "impact_predictions": {
                    "financial": financial_impact,
                    "reputational": reputational_impact,
                    "operational": operational_impact,
                    "regulatory": regulatory_impact,
                    "market": market_impact
                },
                "impact_scenarios": scenarios,
                "overall_impact_score": self._calculate_overall_impact([
                    financial_impact, reputational_impact, operational_impact,
                    regulatory_impact, market_impact
                ])
            }

        except Exception as e:
            logger.error("ImpactPredictionAgent failed", error=str(e))
            return {"error": str(e), "impact_predictions": {}}

    def _predict_financial_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Predict financial impact"""
        company_profile = context.get("company_profile", {})
        external_signals = context.get(
            "social_context", {}).get("external_signals", {})

        # Base impact on market indicators
        stock_movement = abs(external_signals.get(
            "market_indicators", {}).get("stock_movement", 0))
        base_financial_impact = min(1.0, stock_movement * 10)

        # Industry modifier
        high_financial_risk_industries = ["finance", "retail", "technology"]
        industry_modifier = 1.2 if company_profile.get(
            "industry") in high_financial_risk_industries else 1.0

        financial_impact = min(1.0, base_financial_impact * industry_modifier)

        return {
            "probability": 0.7,
            "magnitude": financial_impact,
            "confidence": 0.6,
            "time_horizon_days": 7
        }

    def _predict_reputational_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Predict reputational impact"""
        external_signals = context.get(
            "social_context", {}).get("external_signals", {})
        social_media = external_signals.get("social_media", {})

        # Base on social media sentiment and volume
        mention_volume = social_media.get("mention_volume", 0)
        sentiment_dist = social_media.get("sentiment_distribution", {})
        negative_sentiment = sentiment_dist.get("negative", 0)

        volume_impact = min(1.0, mention_volume / 1000)
        sentiment_impact = negative_sentiment

        reputational_impact = (volume_impact * 0.4 + sentiment_impact * 0.6)

        return {
            "probability": 0.8,
            "magnitude": reputational_impact,
            "confidence": 0.7,
            "time_horizon_days": 14
        }

    def _predict_operational_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Predict operational impact"""
        stakeholders = context.get("relations", [])

        # Count operational stakeholders (partners, suppliers)
        operational_stakeholders = [s for s in stakeholders if s.get("type") in [
            "partner", "supplier"]]
        operational_exposure = min(1.0, len(operational_stakeholders) / 5)

        return {
            "probability": 0.5,
            "magnitude": operational_exposure,
            "confidence": 0.6,
            "time_horizon_days": 3
        }

    def _predict_regulatory_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Predict regulatory impact"""
        company_profile = context.get("company_profile", {})
        stakeholders = context.get("relations", [])

        # High regulatory risk industries
        high_reg_industries = ["finance", "healthcare",
                               "energy", "telecommunications"]
        industry_risk = 0.8 if company_profile.get(
            "industry") in high_reg_industries else 0.3

        # Regulator stakeholder involvement
        regulators = [s for s in stakeholders if s.get("type") == "regulator"]
        regulator_involvement = min(1.0, len(regulators) / 2)

        regulatory_impact = (industry_risk * 0.6 + regulator_involvement * 0.4)

        return {
            "probability": 0.4,
            "magnitude": regulatory_impact,
            "confidence": 0.5,
            "time_horizon_days": 30
        }

    def _predict_market_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Predict market impact"""
        external_signals = context.get(
            "social_context", {}).get("external_signals", {})
        market_indicators = external_signals.get("market_indicators", {})

        stock_movement = abs(market_indicators.get("stock_movement", 0))
        volume_spike = market_indicators.get("volume_spike", False)

        base_impact = stock_movement * 5  # Amplify small movements
        volume_boost = 0.2 if volume_spike else 0

        market_impact = min(1.0, base_impact + volume_boost)

        return {
            "probability": 0.6,
            "magnitude": market_impact,
            "confidence": 0.7,
            "time_horizon_days": 1
        }

    def _generate_impact_scenarios(self, impact_predictions: List[Dict[str, float]]) -> Dict[str, Any]:
        """Generate impact scenarios based on predictions"""
        scenarios = {}

        # Conservative scenario (lower bounds)
        conservative_impacts = [
            p.get("magnitude", 0) * 0.7 for p in impact_predictions]
        scenarios["conservative"] = {
            "overall_impact": sum(conservative_impacts) / len(conservative_impacts),
            "probability": 0.8
        }

        # Realistic scenario (expected values)
        realistic_impacts = [p.get("magnitude", 0) for p in impact_predictions]
        scenarios["realistic"] = {
            "overall_impact": sum(realistic_impacts) / len(realistic_impacts),
            "probability": 0.6
        }

        # Pessimistic scenario (upper bounds)
        pessimistic_impacts = [min(1.0, p.get("magnitude", 0) * 1.3)
                               for p in impact_predictions]
        scenarios["pessimistic"] = {
            "overall_impact": sum(pessimistic_impacts) / len(pessimistic_impacts),
            "probability": 0.2
        }

        return scenarios

    def _calculate_overall_impact(self, impact_predictions: List[Dict[str, float]]) -> float:
        """Calculate weighted overall impact score"""
        weights = [0.25, 0.25, 0.2, 0.15,
                   0.15]  # financial, reputational, operational, regulatory, market

        total_impact = 0
        for i, prediction in enumerate(impact_predictions):
            if i < len(weights):
                magnitude = prediction.get("magnitude", 0)
                probability = prediction.get("probability", 0.5)
                total_impact += magnitude * probability * weights[i]

        return min(1.0, total_impact)


class StakeholderExposureAgent:
    """Stakeholder Risk Specialist - Agent 3/7"""

    def __init__(self, firestore_read_tool: FirestoreReadTool) -> None:
        self.read_tool = firestore_read_tool

    async def execute(self, session: Session) -> Dict[str, Any]:
        """Analyze stakeholder exposure and communication risks"""
        snapshot_data = session.state.get("snapshot_data", {})

        try:
            context = snapshot_data.get("context", {})
            stakeholders = context.get("relations", [])

            # Analyze exposure by stakeholder type
            exposure_analysis = self._analyze_stakeholder_exposure(
                stakeholders)

            # Identify communication priorities
            communication_priorities = self._prioritize_communications(
                stakeholders)

            # Assess relationship impact
            relationship_impact = self._assess_relationship_impact(
                stakeholders)

            return {
                "stakeholder_exposure": exposure_analysis,
                "communication_priorities": communication_priorities,
                "relationship_impact": relationship_impact,
                "total_stakeholders_at_risk": len([s for s in stakeholders if s.get("importance_score", 0) > 0.5])
            }

        except Exception as e:
            logger.error("StakeholderExposureAgent failed", error=str(e))
            return {"error": str(e), "stakeholder_exposure": {}}

    def _analyze_stakeholder_exposure(self, stakeholders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze exposure levels by stakeholder type"""
        exposure_by_type = {}

        for stakeholder in stakeholders:
            stakeholder_type = stakeholder.get("type", "unknown")
            importance = stakeholder.get("importance_score", 0)

            if stakeholder_type not in exposure_by_type:
                exposure_by_type[stakeholder_type] = {
                    "count": 0,
                    "total_exposure": 0,
                    "max_exposure": 0,
                    "stakeholders": []
                }

            exposure_by_type[stakeholder_type]["count"] += 1
            exposure_by_type[stakeholder_type]["total_exposure"] += importance
            exposure_by_type[stakeholder_type]["max_exposure"] = max(
                exposure_by_type[stakeholder_type]["max_exposure"], importance
            )
            exposure_by_type[stakeholder_type]["stakeholders"].append({
                "name": stakeholder.get("name", "Unknown"),
                "exposure": importance
            })

        # Calculate average exposure per type
        for type_data in exposure_by_type.values():
            type_data["avg_exposure"] = type_data["total_exposure"] / \
                type_data["count"]

        return exposure_by_type

    def _prioritize_communications(self, stakeholders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize stakeholder communications based on influence and exposure"""

        # Score stakeholders for communication priority
        scored_stakeholders = []
        for stakeholder in stakeholders:
            importance = stakeholder.get("importance_score", 0)
            stakeholder_type = stakeholder.get("type", "unknown")

            # Type-based priority weights
            type_weights = {
                "customer": 0.9,
                "investor": 0.8,
                "regulator": 0.95,
                "media": 0.85,
                "partner": 0.7
            }

            type_weight = type_weights.get(stakeholder_type, 0.5)
            priority_score = importance * type_weight

            scored_stakeholders.append({
                "name": stakeholder.get("name", "Unknown"),
                "type": stakeholder_type,
                "priority_score": priority_score,
                "importance_score": importance,
                "recommended_timeline": self._recommend_communication_timeline(priority_score),
                "communication_method": self._recommend_communication_method(stakeholder)
            })

        # Sort by priority score
        return sorted(scored_stakeholders, key=lambda x: x["priority_score"], reverse=True)[:10]

    def _recommend_communication_timeline(self, priority_score: float) -> str:
        """Recommend communication timeline based on priority"""
        if priority_score > 0.8:
            return "immediate (within 1 hour)"
        elif priority_score > 0.6:
            return "urgent (within 4 hours)"
        elif priority_score > 0.4:
            return "important (within 24 hours)"
        else:
            return "standard (within 48 hours)"

    def _recommend_communication_method(self, stakeholder: Dict[str, Any]) -> str:
        """Recommend communication method based on stakeholder profile"""
        stakeholder_type = stakeholder.get("type", "unknown")
        importance = stakeholder.get("importance_score", 0)

        if importance > 0.8:
            return "direct_call"
        elif stakeholder_type in ["regulator", "media"]:
            return "formal_statement"
        elif stakeholder_type == "customer":
            return "public_announcement"
        else:
            return "email_notification"

    def _assess_relationship_impact(self, stakeholders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess potential impact on stakeholder relationships"""
        relationship_risks = {}

        for stakeholder in stakeholders:
            stakeholder_type = stakeholder.get("type", "unknown")
            importance = stakeholder.get("importance_score", 0)

            # Communication history analysis
            comm_history = stakeholder.get("communication_history", [])
            recent_communications = len(
                [c for c in comm_history[-5:]])  # Last 5 interactions

            relationship_strength = min(1.0, recent_communications / 5)
            risk_factor = importance * (1 - relationship_strength)

            relationship_risks[stakeholder.get("name", "Unknown")] = {
                "relationship_strength": relationship_strength,
                "risk_factor": risk_factor,
                "recommended_action": "strengthen_relationship" if risk_factor > 0.6 else "maintain_contact"
            }

        return relationship_risks


class TimelineAnalysisAgent:
    """Temporal Dynamics Specialist - Agent 4/7"""

    def __init__(self, firestore_read_tool: FirestoreReadTool) -> None:
        self.read_tool = firestore_read_tool

    async def execute(self, session: Session) -> Dict[str, Any]:
        """Analyze time-sensitive aspects and optimal response windows"""
        snapshot_data = session.state.get("snapshot_data", {})

        try:
            context = snapshot_data.get("context", {})

            # Analyze response urgency
            urgency_analysis = self._analyze_response_urgency(context)

            # Identify escalation triggers
            escalation_triggers = self._identify_escalation_triggers(context)

            # Calculate optimal timing windows
            timing_windows = self._calculate_timing_windows(context)

            return {
                "temporal_analysis": {
                    "response_urgency": urgency_analysis,
                    "escalation_triggers": escalation_triggers,
                    "optimal_timing": timing_windows,
                    "time_pressure_score": self._calculate_time_pressure(context)
                }
            }

        except Exception as e:
            logger.error("TimelineAnalysisAgent failed", error=str(e))
            return {"error": str(e), "temporal_analysis": {}}

    def _analyze_response_urgency(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how urgently response is needed"""
        last_24h = context.get("last_24h_activity", {})
        external_signals = context.get(
            "social_context", {}).get("external_signals", {})

        # Recent activity indicates urgency
        recent_events = last_24h.get("last_24h_events", 0)
        event_urgency = min(1.0, recent_events / 3)

        # Social media velocity
        mention_volume = external_signals.get(
            "social_media", {}).get("mention_volume", 0)
        social_urgency = min(1.0, mention_volume / 500)

        # Market pressure
        market_movement = abs(external_signals.get(
            "market_indicators", {}).get("stock_movement", 0))
        market_urgency = min(1.0, market_movement * 20)

        overall_urgency = max(event_urgency, social_urgency, market_urgency)

        return {
            "overall_urgency_score": overall_urgency,
            "contributing_factors": {
                "recent_events": event_urgency,
                "social_media": social_urgency,
                "market_pressure": market_urgency
            },
            "recommended_response_time": self._recommend_response_time(overall_urgency)
        }

    def _identify_escalation_triggers(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential escalation triggers and thresholds"""
        triggers = []

        external_signals = context.get(
            "social_context", {}).get("external_signals", {})

        # Social media escalation triggers
        current_mentions = external_signals.get(
            "social_media", {}).get("mention_volume", 0)
        triggers.append({
            "type": "social_media_spike",
            "threshold": current_mentions * 2,
            "current_value": current_mentions,
            "risk_level": "high" if current_mentions > 1000 else "medium"
        })

        # Market escalation triggers
        current_movement = abs(external_signals.get(
            "market_indicators", {}).get("stock_movement", 0))
        triggers.append({
            "type": "market_volatility",
            "threshold": max(0.05, current_movement * 2),
            "current_value": current_movement,
            "risk_level": "high" if current_movement > 0.03 else "medium"
        })

        # Stakeholder escalation
        stakeholders = context.get("relations", [])
        high_influence_count = len(
            [s for s in stakeholders if s.get("importance_score", 0) > 0.8])
        triggers.append({
            "type": "stakeholder_engagement",
            "threshold": high_influence_count + 2,
            "current_value": high_influence_count,
            "risk_level": "medium"
        })

        return triggers

    def _calculate_timing_windows(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Calculate optimal timing windows for different actions"""
        urgency_score = self._calculate_time_pressure(context)

        if urgency_score > 0.8:
            return {
                "initial_response": "within 30 minutes",
                "stakeholder_outreach": "within 2 hours",
                "public_statement": "within 4 hours",
                "detailed_plan": "within 8 hours"
            }
        elif urgency_score > 0.6:
            return {
                "initial_response": "within 2 hours",
                "stakeholder_outreach": "within 6 hours",
                "public_statement": "within 12 hours",
                "detailed_plan": "within 24 hours"
            }
        else:
            return {
                "initial_response": "within 4 hours",
                "stakeholder_outreach": "within 12 hours",
                "public_statement": "within 24 hours",
                "detailed_plan": "within 48 hours"
            }

    def _calculate_time_pressure(self, context: Dict[str, Any]) -> float:
        """Calculate overall time pressure score"""
        last_24h = context.get("last_24h_activity", {})
        external_signals = context.get(
            "social_context", {}).get("external_signals", {})

        activity_pressure = min(1.0, last_24h.get("last_24h_events", 0) / 4)
        social_pressure = min(1.0, external_signals.get(
            "social_media", {}).get("mention_volume", 0) / 1000)
        market_pressure = min(1.0, abs(external_signals.get(
            "market_indicators", {}).get("stock_movement", 0)) * 20)

        return max(activity_pressure, social_pressure, market_pressure)

    def _recommend_response_time(self, urgency_score: float) -> str:
        """Recommend response timeframe based on urgency"""
        if urgency_score > 0.8:
            return "immediate (within 1 hour)"
        elif urgency_score > 0.6:
            return "urgent (within 4 hours)"
        elif urgency_score > 0.4:
            return "important (within 12 hours)"
        else:
            return "standard (within 24 hours)"


class CompetitiveContextAgent:
    """Market Position Specialist - Agent 5/7"""

    def __init__(self, firestore_read_tool: FirestoreReadTool) -> None:
        self.read_tool = firestore_read_tool

    async def execute(self, session: Session) -> Dict[str, Any]:
        """Analyze competitive implications and market positioning impact"""
        snapshot_data = session.state.get("snapshot_data", {})

        try:
            context = snapshot_data.get("context", {})
            company_profile = context.get("company_profile", {})
            external_signals = context.get(
                "social_context", {}).get("external_signals", {})

            # Analyze competitive positioning impact
            positioning_impact = self._analyze_positioning_impact(
                company_profile, external_signals)

            # Assess market share risks
            market_share_risk = self._assess_market_share_risk(
                external_signals)

            # Evaluate brand differentiation impact
            brand_impact = self._evaluate_brand_impact(
                company_profile, external_signals)

            return {
                "competitive_analysis": {
                    "positioning_impact": positioning_impact,
                    "market_share_risk": market_share_risk,
                    "brand_differentiation_impact": brand_impact,
                    "competitive_advantage_score": self._calculate_competitive_advantage(
                        positioning_impact, market_share_risk, brand_impact
                    )
                },
                "competitive_recommendations": self._generate_competitive_recommendations(
                    positioning_impact, market_share_risk, brand_impact
                )
            }

        except Exception as e:
            logger.error("CompetitiveContextAgent failed", error=str(e))
            return {"error": str(e), "competitive_analysis": {}}

    def _analyze_positioning_impact(self, company_profile: Dict[str, Any],
                                    external_signals: Dict[str, Any]) -> Dict[str, float]:
        """Analyze impact on market positioning"""
        industry = company_profile.get("industry", "unknown")
        brand_voice = company_profile.get("brand_voice", "")

        # Industry-specific positioning risks
        high_positioning_risk_industries = [
            "technology", "media", "consumer_goods"]
        industry_risk = 0.7 if industry in high_positioning_risk_industries else 0.4

        # Brand voice strength (stronger voice = better crisis resilience)
        # Normalized brand voice strength
        brand_strength = min(1.0, len(brand_voice) / 200)
        positioning_resilience = brand_strength

        # Market sentiment impact
        news_sentiment = external_signals.get(
            "news_sentiment", {}).get("overall_score", 0)
        # Negative sentiment only
        sentiment_impact = abs(news_sentiment) if news_sentiment < 0 else 0

        positioning_risk = (industry_risk * 0.4 + sentiment_impact *
                            0.4 + (1 - positioning_resilience) * 0.2)

        return {
            "positioning_risk": min(1.0, positioning_risk),
            "positioning_resilience": positioning_resilience,
            "industry_risk_factor": industry_risk,
            "sentiment_impact": sentiment_impact
        }

    def _assess_market_share_risk(self, external_signals: Dict[str, Any]) -> Dict[str, float]:
        """Assess potential market share impact"""
        market_indicators = external_signals.get("market_indicators", {})
        competitive_activity = external_signals.get(
            "industry_context", {}).get("competitive_activity", [])

        # Market volatility risk
        stock_movement = abs(market_indicators.get("stock_movement", 0))
        volatility_risk = min(1.0, stock_movement * 15)

        # Competitive pressure
        competitive_pressure = min(1.0, len(competitive_activity) / 3)

        # Calculate overall market share risk
        market_share_risk = (volatility_risk * 0.6 +
                             competitive_pressure * 0.4)

        return {
            "market_share_risk": market_share_risk,
            "volatility_factor": volatility_risk,
            "competitive_pressure": competitive_pressure,
            "protective_actions_needed": market_share_risk > 0.6
        }

    def _evaluate_brand_impact(self, company_profile: Dict[str, Any],
                               external_signals: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate impact on brand differentiation"""
        social_media = external_signals.get("social_media", {})
        sentiment_dist = social_media.get("sentiment_distribution", {})

        # Brand sentiment impact
        negative_sentiment = sentiment_dist.get("negative", 0)
        positive_sentiment = sentiment_dist.get("positive", 0)

        # Net brand impact (negative sentiment hurts, positive helps)
        net_brand_impact = negative_sentiment - \
            (positive_sentiment * 0.5)  # Negative weighs more

        # Brand voice resilience
        brand_voice = company_profile.get("brand_voice", "")
        brand_resilience = min(1.0, len(brand_voice) / 150)

        # Overall brand impact
        brand_impact_score = max(0, net_brand_impact * (1 - brand_resilience))

        return {
            "brand_impact_score": brand_impact_score,
            "negative_sentiment": negative_sentiment,
            "positive_sentiment": positive_sentiment,
            "brand_resilience": brand_resilience,
            "differentiation_at_risk": brand_impact_score > 0.5
        }

    def _calculate_competitive_advantage(self, positioning: Dict[str, float],
                                         market_share: Dict[str, float],
                                         brand: Dict[str, float]) -> float:
        """Calculate overall competitive advantage score"""
        positioning_score = 1 - positioning.get("positioning_risk", 0)
        market_score = 1 - market_share.get("market_share_risk", 0)
        brand_score = 1 - brand.get("brand_impact_score", 0)

        return (positioning_score * 0.4 + market_score * 0.3 + brand_score * 0.3)

    def _generate_competitive_recommendations(self, positioning: Dict[str, float],
                                              market_share: Dict[str, float],
                                              brand: Dict[str, float]) -> List[str]:
        """Generate competitive strategy recommendations"""
        recommendations = []

        if positioning.get("positioning_risk", 0) > 0.6:
            recommendations.append(
                "Reinforce market positioning through targeted messaging")

        if market_share.get("market_share_risk", 0) > 0.6:
            recommendations.append(
                "Implement market share protection strategies")

        if brand.get("brand_impact_score", 0) > 0.5:
            recommendations.append("Execute brand reputation recovery plan")

        if brand.get("differentiation_at_risk", False):
            recommendations.append(
                "Strengthen brand differentiation messaging")

        return recommendations


class LegalComplianceAgent:
    """Regulatory Risk Specialist - Agent 6/7"""

    def __init__(self, firestore_read_tool: FirestoreReadTool) -> None:
        self.read_tool = firestore_read_tool

    async def execute(self, session: Session) -> Dict[str, Any]:
        """Analyze legal implications and regulatory compliance risks"""
        snapshot_data = session.state.get("snapshot_data", {})

        try:
            context = snapshot_data.get("context", {})
            company_profile = context.get("company_profile", {})
            stakeholders = context.get("relations", [])

            # Assess regulatory violations risk
            regulatory_risk = self._assess_regulatory_risk(
                company_profile, stakeholders)

            # Analyze litigation exposure
            litigation_risk = self._analyze_litigation_exposure(context)

            # Identify compliance gaps
            compliance_gaps = self._identify_compliance_gaps(company_profile)

            return {
                "legal_analysis": {
                    "regulatory_risk": regulatory_risk,
                    "litigation_exposure": litigation_risk,
                    "compliance_gaps": compliance_gaps,
                    "overall_legal_risk": self._calculate_overall_legal_risk(
                        regulatory_risk, litigation_risk, compliance_gaps
                    )
                },
                "legal_recommendations": self._generate_legal_recommendations(
                    regulatory_risk, litigation_risk, compliance_gaps
                )
            }

        except Exception as e:
            logger.error("LegalComplianceAgent failed", error=str(e))
            return {"error": str(e), "legal_analysis": {}}

    def _assess_regulatory_risk(self, company_profile: Dict[str, Any],
                                stakeholders: List[Dict[str, Any]]) -> Dict[str, float]:
        """Assess regulatory violation risks"""
        industry = company_profile.get("industry", "unknown")

        # Industry-specific regulatory risk levels
        high_reg_risk_industries = {
            "finance": 0.9,
            "healthcare": 0.8,
            "energy": 0.7,
            "telecommunications": 0.6
        }

        base_regulatory_risk = high_reg_risk_industries.get(industry, 0.3)

        # Regulator stakeholder involvement amplifies risk
        regulators = [s for s in stakeholders if s.get("type") == "regulator"]
        regulator_involvement = min(1.0, len(regulators) / 2)

        regulatory_risk = min(1.0, base_regulatory_risk +
                              (regulator_involvement * 0.3))

        return {
            "regulatory_risk_score": regulatory_risk,
            "industry_base_risk": base_regulatory_risk,
            "regulator_involvement": regulator_involvement,
            "regulators_involved": len(regulators)
        }

    def _analyze_litigation_exposure(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Analyze potential litigation exposure"""
        stakeholders = context.get("relations", [])
        external_signals = context.get(
            "social_context", {}).get("external_signals", {})

        # Customer stakeholder dissatisfaction
        customers = [s for s in stakeholders if s.get("type") == "customer"]
        # More customers = more litigation risk
        customer_risk = len(customers) * 0.1

        # Negative sentiment litigation risk
        sentiment_dist = external_signals.get(
            "social_media", {}).get("sentiment_distribution", {})
        negative_sentiment = sentiment_dist.get("negative", 0)
        sentiment_litigation_risk = negative_sentiment * 0.6

        litigation_exposure = min(
            1.0, customer_risk + sentiment_litigation_risk)

        return {
            "litigation_exposure_score": litigation_exposure,
            "customer_risk_factor": customer_risk,
            "sentiment_risk_factor": sentiment_litigation_risk,
            "high_risk_threshold_exceeded": litigation_exposure > 0.7
        }

    def _identify_compliance_gaps(self, company_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential compliance gaps"""
        gaps = []

        industry = company_profile.get("industry", "unknown")

        # Industry-specific compliance requirements
        if industry == "finance":
            gaps.extend([
                {"area": "financial_reporting", "risk_level": 0.6,
                    "requirement": "SOX compliance"},
                {"area": "data_privacy", "risk_level": 0.5,
                    "requirement": "PCI DSS compliance"}
            ])
        elif industry == "healthcare":
            gaps.extend([
                {"area": "patient_privacy", "risk_level": 0.8,
                    "requirement": "HIPAA compliance"},
                {"area": "data_security", "risk_level": 0.6,
                    "requirement": "HITECH compliance"}
            ])
        elif industry == "technology":
            gaps.extend([
                {"area": "data_privacy", "risk_level": 0.7,
                    "requirement": "GDPR/CCPA compliance"},
                {"area": "platform_safety", "risk_level": 0.5,
                    "requirement": "Platform liability"}
            ])
        else:
            gaps.append({
                "area": "general_compliance",
                "risk_level": 0.4,
                "requirement": "Standard business regulations"
            })

        return gaps

    def _calculate_overall_legal_risk(self, regulatory_risk: Dict[str, float],
                                      litigation_risk: Dict[str, float],
                                      compliance_gaps: List[Dict[str, Any]]) -> float:
        """Calculate overall legal risk score"""
        reg_score = regulatory_risk.get("regulatory_risk_score", 0)
        lit_score = litigation_risk.get("litigation_exposure_score", 0)

        # Compliance gaps contribution
        gap_scores = [gap.get("risk_level", 0) for gap in compliance_gaps]
        avg_gap_score = sum(gap_scores) / len(gap_scores) if gap_scores else 0

        return (reg_score * 0.4 + lit_score * 0.3 + avg_gap_score * 0.3)

    def _generate_legal_recommendations(self, regulatory_risk: Dict[str, float],
                                        litigation_risk: Dict[str, float],
                                        compliance_gaps: List[Dict[str, Any]]) -> List[str]:
        """Generate legal risk mitigation recommendations"""
        recommendations = []

        if regulatory_risk.get("regulatory_risk_score", 0) > 0.6:
            recommendations.append(
                "Engage regulatory affairs team immediately")
            recommendations.append(
                "Prepare regulatory compliance documentation")

        if litigation_risk.get("litigation_exposure_score", 0) > 0.6:
            recommendations.append(
                "Review legal liability and prepare defensive documentation")
            recommendations.append(
                "Consider proactive customer communication to reduce litigation risk")

        if any(gap.get("risk_level", 0) > 0.6 for gap in compliance_gaps):
            recommendations.append("Conduct immediate compliance audit")
            recommendations.append("Implement compliance remediation measures")

        return recommendations


class RiskIntegrationAgent:
    """Multi-dimensional Risk Integration Specialist - Agent 7/7"""

    def __init__(self, firestore_read_tool: FirestoreReadTool) -> None:
        self.read_tool = firestore_read_tool

    async def execute(self, session: Session) -> Dict[str, Any]:
        """Integrate all risk dimensions and identify interdependencies"""

        try:
            # Get all previous analysis results
            analysis_results = session.state.get("analysis_results", {})

            # Extract risk scores from each analysis
            risk_scores = self._extract_risk_scores(analysis_results)

            # Analyze risk interdependencies
            interdependencies = self._analyze_risk_interdependencies(
                risk_scores)

            # Calculate integrated risk profile
            integrated_risk = self._calculate_integrated_risk(
                risk_scores, interdependencies)

            return {
                "risk_integration": {
                    "individual_risks": risk_scores,
                    "risk_interdependencies": interdependencies,
                    "integrated_risk_profile": integrated_risk,
                    "risk_amplification_factors": self._identify_amplification_factors(analysis_results)
                }
            }

        except Exception as e:
            logger.error("RiskIntegrationAgent failed", error=str(e))
            return {"error": str(e), "risk_integration": {}}

    def _extract_risk_scores(self, analysis_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract risk scores from all analysis agents"""
        risk_scores = {}

        # Severity risks
        if "severityassessment" in analysis_results:
            severity_data = analysis_results["severityassessment"].get(
                "data", {})
            risk_scores["severity"] = severity_data.get(
                "severity_analysis", {}).get("overall_severity", 0)

        # Impact risks
        if "impactprediction" in analysis_results:
            impact_data = analysis_results["impactprediction"].get("data", {})
            risk_scores["impact"] = impact_data.get("overall_impact_score", 0)

        # Stakeholder risks
        if "stakeholderexposure" in analysis_results:
            stakeholder_data = analysis_results["stakeholderexposure"].get("data", {
            })
            total_at_risk = stakeholder_data.get(
                "total_stakeholders_at_risk", 0)
            risk_scores["stakeholder"] = min(1.0, total_at_risk / 5)

        # Timeline risks
        if "timelineanalysis" in analysis_results:
            timeline_data = analysis_results["timelineanalysis"].get(
                "data", {})
            risk_scores["timeline"] = timeline_data.get(
                "temporal_analysis", {}).get("time_pressure_score", 0)

        # Competitive risks
        if "competitivecontext" in analysis_results:
            competitive_data = analysis_results["competitivecontext"].get(
                "data", {})
            competitive_score = competitive_data.get(
                "competitive_analysis", {}).get("competitive_advantage_score", 1)
            risk_scores["competitive"] = 1 - \
                competitive_score  # Invert advantage to risk

        # Legal risks
        if "legalcompliance" in analysis_results:
            legal_data = analysis_results["legalcompliance"].get("data", {})
            risk_scores["legal"] = legal_data.get(
                "legal_analysis", {}).get("overall_legal_risk", 0)

        return risk_scores

    def _analyze_risk_interdependencies(self, risk_scores: Dict[str, float]) -> Dict[str, List[str]]:
        """Analyze how different risks amplify each other"""
        interdependencies = {}

        # Legal-Regulatory interdependency
        if risk_scores.get("legal", 0) > 0.6 and risk_scores.get("severity", 0) > 0.6:
            interdependencies["legal_severity_amplification"] = [
                "High legal risk amplifies crisis severity",
                "Regulatory scrutiny increases with crisis magnitude"
            ]

        # Stakeholder-Competitive interdependency
        if risk_scores.get("stakeholder", 0) > 0.6 and risk_scores.get("competitive", 0) > 0.6:
            interdependencies["stakeholder_competitive_amplification"] = [
                "Stakeholder dissatisfaction opens competitive opportunities",
                "Competitive pressure increases stakeholder sensitivity"
            ]

        # Timeline-Impact interdependency
        if risk_scores.get("timeline", 0) > 0.7 and risk_scores.get("impact", 0) > 0.6:
            interdependencies["timeline_impact_amplification"] = [
                "Time pressure amplifies potential impact",
                "High impact situations require faster response"
            ]

        return interdependencies

    def _calculate_integrated_risk(self, risk_scores: Dict[str, float],
                                   interdependencies: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate integrated risk profile considering interdependencies"""

        # Base weighted risk calculation
        risk_weights = {
            "severity": 0.25,
            "impact": 0.25,
            "stakeholder": 0.15,
            "timeline": 0.15,
            "competitive": 0.1,
            "legal": 0.1
        }

        base_risk = sum(
            risk_scores.get(risk_type, 0) * weight
            for risk_type, weight in risk_weights.items()
        )

        # Amplification factor from interdependencies
        # 10% amplification per interdependency
        amplification_factor = 1.0 + (len(interdependencies) * 0.1)

        integrated_risk = min(1.0, base_risk * amplification_factor)

        return {
            "base_risk_score": base_risk,
            "amplification_factor": amplification_factor,
            "integrated_risk_score": integrated_risk,
            "risk_category": self._categorize_risk(integrated_risk),
            "interdependency_count": len(interdependencies)
        }

    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize integrated risk level"""
        if risk_score > 0.8:
            return "critical"
        elif risk_score > 0.6:
            return "high"
        elif risk_score > 0.4:
            return "medium"
        else:
            return "low"

    def _identify_amplification_factors(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Identify factors that amplify overall risk"""
        factors = []

        # Check for high-risk combinations
        severity_high = False
        legal_high = False
        stakeholder_high = False

        if "severityassessment" in analysis_results:
            severity_score = analysis_results["severityassessment"].get(
                "data", {}).get("severity_analysis", {}).get("overall_severity", 0)
            severity_high = severity_score > 0.7

        if "legalcompliance" in analysis_results:
            legal_score = analysis_results["legalcompliance"].get(
                "data", {}).get("legal_analysis", {}).get("overall_legal_risk", 0)
            legal_high = legal_score > 0.6

        if "stakeholderexposure" in analysis_results:
            stakeholder_count = analysis_results["stakeholderexposure"].get(
                "data", {}).get("total_stakeholders_at_risk", 0)
            stakeholder_high = stakeholder_count > 3

        # Identify amplification patterns
        if severity_high and legal_high:
            factors.append(
                "High severity crisis with legal implications creates regulatory scrutiny")

        if stakeholder_high and legal_high:
            factors.append(
                "Multiple at-risk stakeholders increase litigation potential")

        if severity_high and stakeholder_high:
            factors.append(
                "High severity with multiple stakeholders amplifies reputational damage")

        return factors


class ScorecardSynthesizerAgent:
    """Integration and Validation Specialist - Final Synthesizer"""

    def __init__(self, db_pool: FirestoreConnectionPool) -> None:
        self.db_pool = db_pool
        self.write_tool = FirestoreWriteTool(db_pool)

    async def execute(self, session: Session) -> Dict[str, Any]:
        """Synthesize all analysis results into final scorecard"""

        try:
            crisis_case_id = session.state.get("crisis_case_id")
            company_id = session.state.get("company_id")
            analysis_results = session.state.get("analysis_results", {})

            # Extract metrics from all analyses
            scorecard_metrics = self._synthesize_metrics(analysis_results)

            # Identify affected entities
            affected_entities = self._identify_affected_entities(
                analysis_results)

            # Calculate confidence metrics
            confidence_metrics = self._calculate_confidence_metrics(
                analysis_results)

            # Create scorecard
            scorecard = Scorecard(
                crisis_case_id=crisis_case_id,
                metrics=scorecard_metrics,
                affected_entities=affected_entities,
                sub_agent_results=self._clean_sub_agent_results(
                    analysis_results),
                confidence_metrics=confidence_metrics
            )

            # Store scorecard in Firestore as an Artifact under company-scoped Crises
            scorecard_path = f"Company/{company_id}/Crises/{crisis_case_id}/Artifacts/{scorecard.scorecard_id}"
            await self.db_pool.create_document(scorecard_path, scorecard.model_dump())

            # Update crisis case with scorecard reference
            await self.db_pool.update_document(
                f"Company/{company_id}/Crises/{crisis_case_id}",
                {
                    "latest_scorecard_id": scorecard.scorecard_id,
                    "current_status": "classified",
                    "severity_score": scorecard_metrics.severity,
                    "confidence_score": confidence_metrics.get("overall_confidence", 0.5),
                    "primary_class": self._determine_primary_class(scorecard_metrics),
                    "affected_stakeholders": [e.entity_id for e in affected_entities],
                    "updated_at": datetime.utcnow()
                }
            )

            # Update dashboard metrics
            await self._update_dashboard_metrics(company_id, scorecard_metrics)

            logger.info("Scorecard synthesized",
                        crisis_case_id=crisis_case_id, scorecard_id=scorecard.scorecard_id)

            return {
                "scorecard_id": scorecard.scorecard_id,
                "synthesis_status": "completed",
                "confidence_score": confidence_metrics.get("overall_confidence", 0.5)
            }

        except Exception as e:
            logger.error("ScorecardSynthesizerAgent failed", error=str(e))
            return {"error": str(e), "scorecard_id": None}

    def _synthesize_metrics(self, analysis_results: Dict[str, Any]) -> ScorecardMetrics:
        """Synthesize all analysis results into scorecard metrics"""

        # Extract individual scores
        severity = 0.5  # default
        if "severityassessment" in analysis_results:
            severity_data = analysis_results["severityassessment"].get(
                "data", {})
            severity = severity_data.get(
                "severity_analysis", {}).get("overall_severity", 0.5)

        impact = 0.5  # default
        if "impactprediction" in analysis_results:
            impact_data = analysis_results["impactprediction"].get("data", {})
            impact = impact_data.get("overall_impact_score", 0.5)

        # Timeline urgency as speed metric
        speed = 0.5  # default
        if "timelineanalysis" in analysis_results:
            timeline_data = analysis_results["timelineanalysis"].get(
                "data", {})
            speed = timeline_data.get("temporal_analysis", {}).get(
                "time_pressure_score", 0.5)

        # Stakeholder reach
        reach = 0.5  # default
        if "stakeholderexposure" in analysis_results:
            stakeholder_data = analysis_results["stakeholderexposure"].get("data", {
            })
            reach = min(1.0, stakeholder_data.get(
                "total_stakeholders_at_risk", 0) / 5)

        # Reputational risk from competitive analysis
        reputational_risk = 0.5  # default
        if "competitivecontext" in analysis_results:
            competitive_data = analysis_results["competitivecontext"].get(
                "data", {})
            brand_impact = competitive_data.get("competitive_analysis", {}).get(
                "brand_differentiation_impact", {})
            reputational_risk = brand_impact.get("brand_impact_score", 0.5)

        # Legal risk
        legal_risk = 0.5  # default
        if "legalcompliance" in analysis_results:
            legal_data = analysis_results["legalcompliance"].get("data", {})
            legal_risk = legal_data.get("legal_analysis", {}).get(
                "overall_legal_risk", 0.5)

        # Financial risk from impact prediction
        financial_risk = 0.5  # default
        if "impactprediction" in analysis_results:
            impact_data = analysis_results["impactprediction"].get("data", {})
            financial_predictions = impact_data.get("impact_predictions", {})
            financial_risk = financial_predictions.get(
                "financial", {}).get("magnitude", 0.5)

        return ScorecardMetrics(
            severity=severity,
            impact=impact,
            speed=speed,
            reach=reach,
            reputational_risk=reputational_risk,
            legal_risk=legal_risk,
            financial_risk=financial_risk
        )

    def _identify_affected_entities(self, analysis_results: Dict[str, Any]) -> List[AffectedEntity]:
        """Identify entities affected by the crisis"""
        affected_entities = []

        # Extract from stakeholder analysis
        if "stakeholderexposure" in analysis_results:
            stakeholder_data = analysis_results["stakeholderexposure"].get("data", {
            })
            exposure_analysis = stakeholder_data.get(
                "stakeholder_exposure", {})

            for stakeholder_type, type_data in exposure_analysis.items():
                for stakeholder_info in type_data.get("stakeholders", []):
                    affected_entities.append(AffectedEntity(
                        entity_id=f"{stakeholder_type}_{stakeholder_info.get('name', 'unknown').replace(' ', '_')}",
                        relation_score=0.8,  # Assuming high relation if in exposure analysis
                        exposure=stakeholder_info.get("exposure", 0.5)
                    ))

        return affected_entities

    def _calculate_confidence_metrics(self, analysis_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence metrics for the overall analysis"""

        # Count successful vs failed sub-agents
        successful_agents = len(
            [r for r in analysis_results.values() if r.get("status") == "success"])
        total_agents = len(analysis_results)

        # Base confidence on successful agent execution
        execution_confidence = successful_agents / \
            total_agents if total_agents > 0 else 0

        # Data quality confidence (placeholder - would be more sophisticated)
        data_quality_confidence = 0.7  # Assume good data quality

        # Overall confidence
        overall_confidence = (execution_confidence * 0.6 +
                              data_quality_confidence * 0.4)

        return {
            "overall_confidence": overall_confidence,
            "execution_confidence": execution_confidence,
            "data_quality_confidence": data_quality_confidence,
            "successful_analyses": successful_agents,
            "total_analyses": total_agents
        }

    def _clean_sub_agent_results(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and prepare sub-agent results for storage"""
        cleaned_results = {}

        for agent_name, result in analysis_results.items():
            if result.get("status") == "success":
                # Store only the data portion, not the wrapper
                cleaned_results[agent_name] = result.get("data", {})
            else:
                # Store error information
                cleaned_results[agent_name] = {
                    "error": result.get("error", "Unknown error")}

        return cleaned_results

    def _determine_primary_class(self, metrics: ScorecardMetrics) -> str:
        """Determine the primary crisis classification"""

        # Analyze which risk dimension is highest
        risk_scores = {
            "severity_driven": metrics.severity,
            "impact_driven": metrics.impact,
            "stakeholder_driven": metrics.reach,
            "time_critical": metrics.speed,
            "reputational": metrics.reputational_risk,
            "legal_regulatory": metrics.legal_risk,
            "financial": metrics.financial_risk
        }

        # Find the dominant risk type
        primary_risk = max(risk_scores, key=risk_scores.get)

        # Map to business classifications
        class_mapping = {
            "severity_driven": "high_severity_crisis",
            "impact_driven": "high_impact_event",
            "stakeholder_driven": "stakeholder_crisis",
            "time_critical": "rapid_response_required",
            "reputational": "reputation_management",
            "legal_regulatory": "compliance_issue",
            "financial": "financial_crisis"
        }

        return class_mapping.get(primary_risk, "general_crisis")

    async def _update_dashboard_metrics(self, company_id: str, metrics: ScorecardMetrics) -> None:
        """Update company dashboard with new crisis metrics"""
        try:
            # Determine if this is a critical crisis
            is_critical = metrics.severity > 0.8 or metrics.impact > 0.8

            # Update dashboard counters
            counter_updates = {
                "num_crises_total": 1,
                "num_active": 1
            }

            if is_critical:
                counter_updates["num_critical"] = 1

            await self.write_tool.update_counters(
                f"dashboards/{company_id}",
                counter_updates,
                company_id
            )

            logger.debug("Dashboard metrics updated",
                         company_id=company_id, is_critical=is_critical)

        except Exception as e:
            logger.warning("Dashboard update failed",
                           company_id=company_id, error=str(e))


# Export
__all__ = ['ClassificationAgent']
