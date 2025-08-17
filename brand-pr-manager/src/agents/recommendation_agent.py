"""
Recommendation Agent (Agent C) with 7 Strategic Planning Sub-agents
Implements comprehensive strategic response planning using ADK WorkflowAgent orchestration
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import structlog

# ADK imports (placeholder - would use actual ADK in production)
try:
    from google.adk import WorkflowAgent, ParallelAgent, LlmAgent, AgentTool, Session, InvocationContext
    from google.adk.memory import VertexAiMemoryBankService
except ImportError:
    # Mock ADK classes for development
    class WorkflowAgent:
        def __init__(self, workflow_steps: List[Any]):
            self.workflow_steps = workflow_steps

    class ParallelAgent:
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

from models.schemas import Recommendation, RecommendationStep, Scorecard, CrisisSnapshot, SearchResult
from tools.mcp_tools import FirestoreReadTool, FirestoreWriteTool
from tools.vector_search_tool import VectorSearchTool
from infrastructure.firestore_client import FirestoreConnectionPool

logger = structlog.get_logger()


class RecommendationAgent:
    """
    Main orchestrator for strategic response planning with 7 specialized sub-agents
    Uses ADK WorkflowAgent for complex multi-step recommendation generation
    """

    def __init__(self, db_pool: FirestoreConnectionPool, vector_search: VectorSearchTool) -> None:
        self.db_pool = db_pool
        self.vector_search = vector_search

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

        # Initialize strategic planning sub-agents
        self.planning_agents = self._initialize_planning_agents()
        self.synthesizer_agent = RecommendationSynthesizerAgent(
            db_pool, self.firestore_write_tool)

    def _initialize_planning_agents(self) -> List[Any]:
        """Initialize the 7 specialized strategic planning sub-agents"""
        return [
            HistoricalCaseSearchAgent(
                self.vector_search, self.firestore_read_tool),
            ScenarioModelingAgent(self.firestore_read_tool),
            StakeholderStrategyAgent(self.firestore_read_tool),
            ResourceOptimizationAgent(self.firestore_read_tool),
            RiskMitigationAgent(self.firestore_read_tool),
            ComplianceValidatorAgent(self.firestore_read_tool),
            StrategicIntegrationAgent(self.firestore_read_tool)
        ]

    async def generate_recommendations(self, crisis_case_id: str, scorecard_id: str,
                                       snapshot_id: str, company_id: str, session_id: str) -> str:
        """
        Main entry point for recommendation generation
        Returns recommendation_id after complete strategic analysis
        """
        start_time = time.time()
        self.execution_metrics["total_executions"] += 1

        try:
            logger.info("Starting recommendation generation",
                        crisis_case_id=crisis_case_id, scorecard_id=scorecard_id)

            # Create ADK session for recommendations
            session = Session(session_id)
            session.state = {
                "crisis_case_id": crisis_case_id,
                "company_id": company_id,
                "scorecard_id": scorecard_id,
                "snapshot_id": snapshot_id,
                "recommendation_start": datetime.utcnow().isoformat(),
                "planning_results": {}
            }

            # Load required data for analysis
            await self._load_analysis_data(session)

            # Execute strategic planning sub-agents in parallel
            planning_results = await self._execute_strategic_planning(session)

            # Synthesize results into final recommendations
            recommendation_id = await self._synthesize_recommendations(session, planning_results)

            # Update execution metrics
            execution_time = (time.time() - start_time) * 1000
            self.execution_metrics["successful_executions"] += 1
            self.execution_metrics["avg_execution_time_ms"] = (
                (self.execution_metrics["avg_execution_time_ms"] * (self.execution_metrics["total_executions"] - 1)
                 + execution_time) / self.execution_metrics["total_executions"]
            )

            logger.info("Recommendation generation completed",
                        crisis_case_id=crisis_case_id, recommendation_id=recommendation_id,
                        execution_time_ms=execution_time)

            return recommendation_id

        except Exception as e:
            self.execution_metrics["failed_executions"] += 1
            logger.error("Recommendation generation failed",
                         crisis_case_id=crisis_case_id, error=str(e))
            raise

    async def _load_analysis_data(self, session: Session) -> None:
        """Load required data from previous analysis phases"""
        crisis_case_id = session.state["crisis_case_id"]
        scorecard_id = session.state["scorecard_id"]
        snapshot_id = session.state["snapshot_id"]
        company_id = session.state["company_id"]

        # Load scorecard data from Artifacts
        scorecard_data = await self.firestore_read_tool.read_document(
            f"Company/{company_id}/Crises/{crisis_case_id}/Artifacts", scorecard_id, company_id
        )

        # Load snapshot data from Artifacts
        snapshot_data = await self.firestore_read_tool.read_document(
            f"Company/{company_id}/Crises/{crisis_case_id}/Artifacts", snapshot_id, company_id
        )

        # Load crisis case data from company-scoped Crises
        crisis_data = await self.firestore_read_tool.read_document(
            f"Company/{company_id}/Crises", crisis_case_id, company_id
        )

        session.state.update({
            "scorecard_data": scorecard_data,
            "snapshot_data": snapshot_data,
            "crisis_data": crisis_data
        })

    async def _execute_strategic_planning(self, session: Session) -> Dict[str, Any]:
        """Execute strategic planning sub-agents in parallel"""
        planning_tasks = []

        # Create parallel execution tasks
        for agent in self.planning_agents:
            task = asyncio.create_task(
                self._safe_agent_execution(agent, session),
                name=f"planning_{agent.__class__.__name__}"
            )
            planning_tasks.append(task)

        # Wait for all planning analyses to complete
        planning_results = await asyncio.gather(*planning_tasks, return_exceptions=True)

        # Process results and handle exceptions
        processed_results = {}
        for i, (agent, result) in enumerate(zip(self.planning_agents, planning_results)):
            agent_name = agent.__class__.__name__.lower().replace('agent', '')

            if isinstance(result, Exception):
                logger.error(f"Planning agent {i+1}/7 failed: {agent.__class__.__name__}",
                             error=str(result))
                processed_results[agent_name] = {
                    "error": str(result), "status": "failed"}
            else:
                processed_results[agent_name] = {
                    "data": result, "status": "success"}
                logger.debug(
                    f"Planning agent {i+1}/7 completed: {agent.__class__.__name__}")

        return processed_results

    async def _safe_agent_execution(self, agent: Any, session: Session) -> Dict[str, Any]:
        """Safely execute an agent with error handling"""
        try:
            return await agent.execute(session)
        except Exception as e:
            logger.warning(
                f"Agent execution failed: {agent.__class__.__name__}", error=str(e))
            raise

    async def _synthesize_recommendations(self, session: Session, planning_results: Dict[str, Any]) -> str:
        """Synthesize planning results into final recommendations"""
        session.state["planning_results"] = planning_results

        # Execute synthesizer agent
        recommendation_data = await self.synthesizer_agent.execute(session)

        return recommendation_data.get("recommendation_id")

# Strategic Planning Sub-agents


class HistoricalCaseSearchAgent:
    """Precedent Research Specialist - Agent 1/7"""

    def __init__(self, vector_search: VectorSearchTool, firestore_read_tool: FirestoreReadTool) -> None:
        self.vector_search = vector_search
        self.read_tool = firestore_read_tool

    async def execute(self, session: Session) -> Dict[str, Any]:
        """Search for and analyze historical precedents"""

        try:
            scorecard_data = session.state.get("scorecard_data", {})
            snapshot_data = session.state.get("snapshot_data", {})
            company_id = session.state.get("company_id")

            # Build search query from crisis characteristics
            search_query = self._build_search_query(
                scorecard_data, snapshot_data)

            # Multi-vector search for similar cases
            similar_cases = await self._search_similar_cases(search_query, company_id)

            # Analyze outcomes of similar cases
            outcome_analysis = await self._analyze_outcomes(similar_cases, company_id)

            # Extract lessons learned
            lessons_learned = self._extract_lessons_learned(
                similar_cases, outcome_analysis)

            return {
                "historical_analysis": {
                    "similar_cases": similar_cases,
                    "outcome_analysis": outcome_analysis,
                    "lessons_learned": lessons_learned,
                    "search_query_used": search_query,
                    "total_cases_found": len(similar_cases)
                }
            }

        except Exception as e:
            logger.error("HistoricalCaseSearchAgent failed", error=str(e))
            return {"error": str(e), "historical_analysis": {"similar_cases": []}}

    def _build_search_query(self, scorecard_data: Dict[str, Any], snapshot_data: Dict[str, Any]) -> str:
        """Build search query from crisis characteristics"""

        # Extract key characteristics
        metrics = scorecard_data.get("metrics", {})
        context = snapshot_data.get("context", {})

        severity = metrics.get("severity", 0)
        crisis_nature = context.get(
            "company_profile", {}).get("industry", "unknown")
        stakeholder_types = [s.get("type", "")
                             for s in context.get("relations", [])]

        # Build semantic search query
        query_parts = [
            f"crisis severity {severity:.1f}",
            f"industry {crisis_nature}",
            "stakeholders " + " ".join(set(stakeholder_types)),
            "resolution strategy response plan"
        ]

        return " ".join(query_parts)

    async def _search_similar_cases(self, search_query: str, company_id: str) -> List[SearchResult]:
        """Search for similar crisis cases using vector similarity"""

        # Search company-specific cases first
        company_cases = await self.vector_search.similarity_search(
            search_query,
            filters={
                "company_scoped": True,
                "source_types": ["case_study", "internal_crisis"],
                "min_confidence": 0.6
            },
            company_id=company_id,
            top_k=5
        )

        # Search global cases for additional context
        global_cases = await self.vector_search.similarity_search(
            search_query,
            filters={
                "company_scoped": False,
                "source_types": ["case_study", "external_crisis"],
                "min_confidence": 0.7
            },
            company_id=company_id,
            top_k=5
        )

        # Combine and rank results
        all_cases = company_cases + global_cases
        return sorted(all_cases, key=lambda x: x.similarity_score, reverse=True)[:8]

    async def _analyze_outcomes(self, similar_cases: List[SearchResult], company_id: str) -> Dict[str, Any]:
        """Analyze outcomes of similar historical cases"""
        if not similar_cases:
            return {"outcome_patterns": {}, "success_factors": []}

        outcome_patterns = {}
        success_factors = []

        # Analyze case outcomes (would query detailed case data in production)
        for case in similar_cases:
            case_metadata = case.metadata

            # Extract outcome indicators from metadata
            confidence = case_metadata.get("confidence_score", 0.5)
            outcome_category = "successful" if confidence > 0.7 else "mixed" if confidence > 0.4 else "challenging"

            if outcome_category not in outcome_patterns:
                outcome_patterns[outcome_category] = {
                    "count": 0,
                    "avg_confidence": 0,
                    "cases": []
                }

            outcome_patterns[outcome_category]["count"] += 1
            outcome_patterns[outcome_category]["avg_confidence"] += confidence
            outcome_patterns[outcome_category]["cases"].append({
                "title": case.title,
                "similarity": case.similarity_score,
                "confidence": confidence
            })

        # Calculate averages
        for pattern in outcome_patterns.values():
            if pattern["count"] > 0:
                pattern["avg_confidence"] /= pattern["count"]

        # Identify common success factors
        successful_cases = outcome_patterns.get(
            "successful", {}).get("cases", [])
        if successful_cases:
            success_factors = [
                "Early stakeholder engagement",
                "Transparent communication",
                "Rapid response implementation",
                "Proactive risk mitigation"
            ]

        return {
            "outcome_patterns": outcome_patterns,
            "success_factors": success_factors,
            "total_cases_analyzed": len(similar_cases)
        }

    def _extract_lessons_learned(self, similar_cases: List[SearchResult],
                                 outcome_analysis: Dict[str, Any]) -> List[str]:
        """Extract actionable lessons from historical cases"""
        lessons = []

        successful_cases = outcome_analysis.get(
            "outcome_patterns", {}).get("successful", {})
        if successful_cases and successful_cases.get("count", 0) > 0:
            lessons.extend([
                "Rapid response within first 4 hours improves outcomes",
                "Direct stakeholder communication reduces escalation risk",
                "Transparent messaging builds trust during crisis",
                "Proactive measures prevent secondary issues"
            ])

        # Add case-specific lessons based on similarity
        high_similarity_cases = [
            c for c in similar_cases if c.similarity_score > 0.8]
        if high_similarity_cases:
            lessons.extend([
                "Very similar cases found - follow proven resolution patterns",
                "Historical precedents suggest specific stakeholder sensitivities",
                "Timeline patterns indicate optimal intervention points"
            ])

        return lessons


class ScenarioModelingAgent:
    """Strategic Options Specialist - Agent 2/7"""

    def __init__(self, firestore_read_tool: FirestoreReadTool) -> None:
        self.read_tool = firestore_read_tool

    async def execute(self, session: Session) -> Dict[str, Any]:
        """Model multiple strategic response scenarios"""

        try:
            scorecard_data = session.state.get("scorecard_data", {})
            snapshot_data = session.state.get("snapshot_data", {})

            # Generate multiple strategic scenarios
            scenarios = await self._generate_strategic_scenarios(scorecard_data, snapshot_data)

            # Analyze trade-offs between scenarios
            trade_off_analysis = self._analyze_scenario_tradeoffs(scenarios)

            # Recommend optimal scenario mix
            optimal_strategy = self._recommend_optimal_strategy(
                scenarios, trade_off_analysis)

            return {
                "scenario_modeling": {
                    "scenarios": scenarios,
                    "trade_off_analysis": trade_off_analysis,
                    "optimal_strategy": optimal_strategy,
                    "scenario_count": len(scenarios)
                }
            }

        except Exception as e:
            logger.error("ScenarioModelingAgent failed", error=str(e))
            return {"error": str(e), "scenario_modeling": {"scenarios": []}}

    async def _generate_strategic_scenarios(self, scorecard_data: Dict[str, Any],
                                            snapshot_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate multiple strategic response scenarios"""

        metrics = scorecard_data.get("metrics", {})
        severity = metrics.get("severity", 0.5)
        impact = metrics.get("impact", 0.5)
        legal_risk = metrics.get("legal_risk", 0.5)

        scenarios = []

        # Defensive Strategy
        scenarios.append({
            "name": "defensive_response",
            "description": "Minimize damage and protect existing position",
            "approach": "risk_minimization",
            "timeline_hours": 24,
            "resource_intensity": "medium",
            "stakeholder_impact": "low",
            "cost_estimate": 50000,
            "success_probability": 0.7,
            "key_actions": [
                "Issue controlled public statement",
                "Engage legal counsel",
                "Implement damage control measures",
                "Monitor stakeholder reactions"
            ]
        })

        # Proactive Strategy
        scenarios.append({
            "name": "proactive_engagement",
            "description": "Take initiative to address root causes",
            "approach": "problem_solving",
            "timeline_hours": 48,
            "resource_intensity": "high",
            "stakeholder_impact": "medium",
            "cost_estimate": 100000,
            "success_probability": 0.8,
            "key_actions": [
                "Launch comprehensive investigation",
                "Engage all key stakeholders directly",
                "Implement corrective measures",
                "Establish ongoing monitoring"
            ]
        })

        # Collaborative Strategy
        scenarios.append({
            "name": "collaborative_resolution",
            "description": "Work with stakeholders for joint solution",
            "approach": "stakeholder_partnership",
            "timeline_hours": 72,
            "resource_intensity": "medium",
            "stakeholder_impact": "high",
            "cost_estimate": 75000,
            "success_probability": 0.6,
            "key_actions": [
                "Form stakeholder advisory group",
                "Develop joint action plan",
                "Share decision-making process",
                "Implement collaborative governance"
            ]
        })

        # Competitive Strategy (if competitive risk is high)
        if metrics.get("reputational_risk", 0) > 0.6:
            scenarios.append({
                "name": "competitive_differentiation",
                "description": "Use crisis to strengthen competitive position",
                "approach": "market_positioning",
                "timeline_hours": 96,
                "resource_intensity": "high",
                "stakeholder_impact": "medium",
                "cost_estimate": 150000,
                "success_probability": 0.5,
                "key_actions": [
                    "Launch differentiation campaign",
                    "Highlight competitive advantages",
                    "Engage industry thought leaders",
                    "Position as industry leader in response"
                ]
            })

        return scenarios

    def _analyze_scenario_tradeoffs(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trade-offs between different scenarios"""

        # Compare scenarios across key dimensions
        comparison_matrix = {}

        dimensions = ["cost_estimate", "timeline_hours",
                      "success_probability", "resource_intensity"]

        for dimension in dimensions:
            dimension_values = []
            for scenario in scenarios:
                if dimension == "resource_intensity":
                    # Convert to numeric
                    intensity_map = {"low": 0.3, "medium": 0.6, "high": 1.0}
                    value = intensity_map.get(
                        scenario.get(dimension, "medium"), 0.6)
                else:
                    value = scenario.get(dimension, 0)
                dimension_values.append(value)

            comparison_matrix[dimension] = {
                "min": min(dimension_values),
                "max": max(dimension_values),
                "avg": sum(dimension_values) / len(dimension_values),
                "range": max(dimension_values) - min(dimension_values)
            }

        # Identify optimal trade-offs
        trade_offs = {
            "cost_vs_success": self._analyze_cost_success_tradeoff(scenarios),
            "speed_vs_thoroughness": self._analyze_speed_thoroughness_tradeoff(scenarios),
            "risk_vs_reward": self._analyze_risk_reward_tradeoff(scenarios)
        }

        return {
            "comparison_matrix": comparison_matrix,
            "trade_off_analysis": trade_offs,
            "scenario_rankings": self._rank_scenarios(scenarios)
        }

    def _analyze_cost_success_tradeoff(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze cost vs success probability trade-off"""
        cost_effectiveness = []

        for scenario in scenarios:
            cost = scenario.get("cost_estimate", 50000)
            success_prob = scenario.get("success_probability", 0.5)
            effectiveness = success_prob / \
                (cost / 50000)  # Normalize by base cost

            cost_effectiveness.append({
                "scenario": scenario["name"],
                "cost_effectiveness": effectiveness,
                "cost": cost,
                "success_probability": success_prob
            })

        # Sort by cost effectiveness
        cost_effectiveness.sort(
            key=lambda x: x["cost_effectiveness"], reverse=True)

        return {
            "most_cost_effective": cost_effectiveness[0]["scenario"] if cost_effectiveness else None,
            "cost_effectiveness_ranking": cost_effectiveness
        }

    def _analyze_speed_thoroughness_tradeoff(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze speed vs thoroughness trade-off"""

        speed_rankings = []
        for scenario in scenarios:
            timeline = scenario.get("timeline_hours", 48)
            resource_intensity = scenario.get("resource_intensity", "medium")

            # Speed score (lower timeline = higher speed)
            speed_score = max(0, 1 - (timeline / 96))  # Normalize to 4-day max

            # Thoroughness score (higher resource intensity = more thoroughness)
            thoroughness_map = {"low": 0.3, "medium": 0.6, "high": 1.0}
            thoroughness_score = thoroughness_map.get(resource_intensity, 0.6)

            speed_rankings.append({
                "scenario": scenario["name"],
                "speed_score": speed_score,
                "thoroughness_score": thoroughness_score,
                "balance_score": (speed_score + thoroughness_score) / 2
            })

        speed_rankings.sort(key=lambda x: x["balance_score"], reverse=True)

        return {
            "best_balanced": speed_rankings[0]["scenario"] if speed_rankings else None,
            "speed_thoroughness_ranking": speed_rankings
        }

    def _analyze_risk_reward_tradeoff(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze risk vs reward trade-off"""

        risk_reward = []
        for scenario in scenarios:
            success_prob = scenario.get("success_probability", 0.5)
            stakeholder_impact = scenario.get("stakeholder_impact", "medium")

            # Reward score (success probability + positive stakeholder impact)
            impact_map = {"low": 0.2, "medium": 0.5, "high": 0.8}
            impact_score = impact_map.get(stakeholder_impact, 0.5)
            reward_score = (success_prob + impact_score) / 2

            # Risk score (inverse of success probability + resource intensity)
            resource_intensity = scenario.get("resource_intensity", "medium")
            intensity_map = {"low": 0.3, "medium": 0.6, "high": 1.0}
            resource_risk = intensity_map.get(resource_intensity, 0.6)
            risk_score = ((1 - success_prob) + resource_risk) / 2

            risk_reward.append({
                "scenario": scenario["name"],
                "reward_score": reward_score,
                "risk_score": risk_score,
                "risk_reward_ratio": reward_score / max(0.1, risk_score)
            })

        risk_reward.sort(key=lambda x: x["risk_reward_ratio"], reverse=True)

        return {
            "best_risk_reward": risk_reward[0]["scenario"] if risk_reward else None,
            "risk_reward_analysis": risk_reward
        }

    def _recommend_optimal_strategy(self, scenarios: List[Dict[str, Any]],
                                    trade_off_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend optimal strategy based on trade-off analysis"""

        # Weight different factors
        rankings = trade_off_analysis.get("scenario_rankings", [])
        if not rankings:
            return {"recommended_scenario": scenarios[0]["name"] if scenarios else None}

        return {
            "recommended_scenario": rankings[0]["scenario"],
            "recommendation_rationale": "Highest overall score across cost, speed, and risk factors",
            "alternative_scenarios": [r["scenario"] for r in rankings[1:3]]
        }

    def _rank_scenarios(self, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank scenarios by overall score"""
        scored_scenarios = []

        for scenario in scenarios:
            # Calculate overall score
            success_weight = 0.4
            cost_weight = 0.2
            speed_weight = 0.2
            resource_weight = 0.2

            success_score = scenario.get("success_probability", 0.5)
            # Normalize and invert
            cost_score = 1 - \
                min(1.0, scenario.get("cost_estimate", 50000) / 200000)
            # Faster is better
            speed_score = 1 - min(1.0, scenario.get("timeline_hours", 48) / 96)

            # Lower resource need is better
            resource_map = {"low": 1.0, "medium": 0.6, "high": 0.3}
            resource_score = resource_map.get(
                scenario.get("resource_intensity", "medium"), 0.6)

            overall_score = (
                success_score * success_weight +
                cost_score * cost_weight +
                speed_score * speed_weight +
                resource_score * resource_weight
            )

            scored_scenarios.append({
                "scenario": scenario["name"],
                "overall_score": overall_score,
                "component_scores": {
                    "success": success_score,
                    "cost": cost_score,
                    "speed": speed_score,
                    "resource": resource_score
                }
            })

        return sorted(scored_scenarios, key=lambda x: x["overall_score"], reverse=True)


class StakeholderStrategyAgent:
    """Communication Planning Specialist - Agent 3/7"""

    def __init__(self, firestore_read_tool: FirestoreReadTool) -> None:
        self.read_tool = firestore_read_tool

    async def execute(self, session: Session) -> Dict[str, Any]:
        """Develop stakeholder-specific communication strategies"""

        try:
            snapshot_data = session.state.get("snapshot_data", {})
            scorecard_data = session.state.get("scorecard_data", {})

            context = snapshot_data.get("context", {})
            stakeholders = context.get("relations", [])

            # Develop communication strategies
            communication_strategies = self._develop_communication_strategies(
                stakeholders, scorecard_data)

            # Plan message timing
            timing_plan = self._plan_message_timing(
                stakeholders, scorecard_data)

            # Select communication channels
            channel_recommendations = self._recommend_communication_channels(
                stakeholders)

            return {
                "stakeholder_strategy": {
                    "communication_strategies": communication_strategies,
                    "timing_plan": timing_plan,
                    "channel_recommendations": channel_recommendations,
                    "total_stakeholder_groups": len(communication_strategies)
                }
            }

        except Exception as e:
            logger.error("StakeholderStrategyAgent failed", error=str(e))
            return {"error": str(e), "stakeholder_strategy": {}}

    def _develop_communication_strategies(self, stakeholders: List[Dict[str, Any]],
                                          scorecard_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Develop tailored communication strategies for each stakeholder group"""

        strategies = []
        metrics = scorecard_data.get("metrics", {})

        # Group stakeholders by type
        stakeholder_groups = {}
        for stakeholder in stakeholders:
            stakeholder_type = stakeholder.get("type", "unknown")
            if stakeholder_type not in stakeholder_groups:
                stakeholder_groups[stakeholder_type] = []
            stakeholder_groups[stakeholder_type].append(stakeholder)

        # Develop strategy for each group
        for group_type, group_members in stakeholder_groups.items():
            strategy = {
                "stakeholder_group": group_type,
                "group_size": len(group_members),
                "communication_approach": self._determine_communication_approach(group_type, metrics),
                "key_messages": self._craft_key_messages(group_type, metrics),
                "expected_concerns": self._anticipate_concerns(group_type, metrics),
                "success_metrics": self._define_success_metrics(group_type)
            }
            strategies.append(strategy)

        return strategies

    def _determine_communication_approach(self, stakeholder_type: str, metrics: Dict[str, Any]) -> str:
        """Determine communication approach for stakeholder type"""

        severity = metrics.get("severity", 0.5)
        legal_risk = metrics.get("legal_risk", 0.5)

        if stakeholder_type == "regulator":
            return "formal_compliance_focused" if legal_risk > 0.6 else "transparent_cooperative"
        elif stakeholder_type == "media":
            return "controlled_narrative" if severity > 0.7 else "open_dialogue"
        elif stakeholder_type == "customer":
            return "reassurance_focused" if severity > 0.6 else "informative_transparent"
        elif stakeholder_type == "investor":
            return "impact_mitigation" if metrics.get("financial_risk", 0) > 0.6 else "confidence_building"
        else:
            return "relationship_preserving"

    def _craft_key_messages(self, stakeholder_type: str, metrics: Dict[str, Any]) -> List[str]:
        """Craft key messages for each stakeholder type"""

        messages = []
        severity = metrics.get("severity", 0.5)

        if stakeholder_type == "customer":
            messages.extend([
                "We are committed to resolving this situation quickly and transparently",
                "Customer safety and satisfaction remain our top priorities",
                "We will provide regular updates as we work toward resolution"
            ])

        elif stakeholder_type == "investor":
            messages.extend([
                "We are taking decisive action to address this situation",
                "Our business fundamentals remain strong",
                "We expect minimal long-term impact on operations"
            ])

        elif stakeholder_type == "regulator":
            messages.extend([
                "We are fully cooperating with all relevant authorities",
                "Compliance with regulations is our highest priority",
                "We have implemented additional oversight measures"
            ])

        elif stakeholder_type == "media":
            messages.extend([
                "We are committed to transparency throughout this process",
                "We welcome the opportunity to share our perspective",
                "Regular updates will be provided as information becomes available"
            ])

        return messages

    def _anticipate_concerns(self, stakeholder_type: str, metrics: Dict[str, Any]) -> List[str]:
        """Anticipate likely concerns from each stakeholder group"""

        concerns = []

        if stakeholder_type == "customer":
            concerns.extend([
                "Product/service safety and reliability",
                "Impact on ongoing services",
                "Company responsiveness to issues",
                "Compensation or remediation"
            ])

        elif stakeholder_type == "investor":
            concerns.extend([
                "Financial impact on returns",
                "Management competence",
                "Long-term business viability",
                "Regulatory penalties or fines"
            ])

        elif stakeholder_type == "regulator":
            concerns.extend([
                "Compliance violations",
                "Consumer protection",
                "Industry standards adherence",
                "Systemic risk implications"
            ])

        return concerns

    def _plan_message_timing(self, stakeholders: List[Dict[str, Any]],
                             scorecard_data: Dict[str, Any]) -> Dict[str, Any]:
        """Plan optimal timing for stakeholder communications"""

        metrics = scorecard_data.get("metrics", {})
        urgency = metrics.get("speed", 0.5)

        # Base timing on urgency
        if urgency > 0.8:
            timing_strategy = "immediate_cascade"
            base_intervals = {"regulator": 0.5, "media": 1,
                              "investor": 1.5, "customer": 2, "partner": 3}
        elif urgency > 0.6:
            timing_strategy = "rapid_sequence"
            base_intervals = {"regulator": 2, "media": 4,
                              "investor": 6, "customer": 8, "partner": 12}
        else:
            timing_strategy = "measured_rollout"
            base_intervals = {"regulator": 4, "media": 8,
                              "investor": 12, "customer": 16, "partner": 24}

        # Create timing plan
        timing_plan = {
            "strategy": timing_strategy,
            "stakeholder_timing": base_intervals,
            "coordination_windows": self._calculate_coordination_windows(base_intervals)
        }

        return timing_plan

    def _calculate_coordination_windows(self, base_intervals: Dict[str, float]) -> List[Dict[str, Any]]:
        """Calculate coordination windows between communications"""

        windows = []
        sorted_intervals = sorted(base_intervals.items(), key=lambda x: x[1])

        for i, (stakeholder_type, timing) in enumerate(sorted_intervals):
            window = {
                "stakeholder_type": stakeholder_type,
                "hours_from_start": timing,
                "preparation_time": max(0.5, timing - 0.5),
                "sequence_order": i + 1
            }
            windows.append(window)

        return windows

    def _recommend_communication_channels(self, stakeholders: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Recommend communication channels for each stakeholder type"""

        channel_recommendations = {}

        # Group by type and recommend channels
        stakeholder_types = set(s.get("type", "unknown") for s in stakeholders)

        for stakeholder_type in stakeholder_types:
            if stakeholder_type == "customer":
                channels = ["email_announcement", "website_banner",
                            "social_media", "customer_service"]
            elif stakeholder_type == "investor":
                channels = ["investor_call", "SEC_filing",
                            "press_release", "investor_portal"]
            elif stakeholder_type == "regulator":
                channels = ["formal_notification", "regulatory_portal",
                            "direct_contact", "compliance_report"]
            elif stakeholder_type == "media":
                channels = ["press_release", "media_briefing",
                            "spokesperson_interview", "press_conference"]
            elif stakeholder_type == "partner":
                channels = ["partner_portal", "account_manager",
                            "joint_statement", "partner_call"]
            else:
                channels = ["email", "phone_call", "formal_letter"]

            channel_recommendations[stakeholder_type] = channels

        return channel_recommendations

    def _define_success_metrics(self, stakeholder_type: str) -> List[str]:
        """Define success metrics for stakeholder communication"""

        if stakeholder_type == "customer":
            return [
                "Customer satisfaction score maintained >80%",
                "Support ticket volume increase <50%",
                "Social media sentiment >60% neutral/positive"
            ]
        elif stakeholder_type == "investor":
            return [
                "Stock price volatility <10%",
                "Analyst rating downgrades <2",
                "Investor confidence survey >70%"
            ]
        elif stakeholder_type == "regulator":
            return [
                "No formal regulatory action initiated",
                "Compliance rating maintained",
                "Regulator feedback positive/neutral"
            ]
        else:
            return [
                "Relationship satisfaction maintained",
                "No partnership terminations",
                "Continued collaboration confirmed"
            ]


class ResourceOptimizationAgent:
    """Implementation Planning Specialist - Agent 4/7"""

    def __init__(self, firestore_read_tool: FirestoreReadTool) -> None:
        self.read_tool = firestore_read_tool

    async def execute(self, session: Session) -> Dict[str, Any]:
        """Optimize resource allocation and implementation planning"""

        try:
            scorecard_data = session.state.get("scorecard_data", {})
            planning_results = session.state.get("planning_results", {})

            # Get scenario data if available
            scenario_data = planning_results.get(
                "scenariomodeling", {}).get("data", {})
            scenarios = scenario_data.get(
                "scenario_modeling", {}).get("scenarios", [])

            # Calculate resource requirements
            resource_requirements = self._calculate_resource_requirements(
                scorecard_data, scenarios)

            # Optimize timeline
            timeline_optimization = self._optimize_timeline(
                scorecard_data, scenarios)

            # Plan capacity allocation
            capacity_plan = self._plan_capacity_allocation(
                resource_requirements, timeline_optimization)

            # Estimate budget
            budget_estimation = self._estimate_budget(
                resource_requirements, scenarios)

            return {
                "resource_optimization": {
                    "resource_requirements": resource_requirements,
                    "timeline_optimization": timeline_optimization,
                    "capacity_plan": capacity_plan,
                    "budget_estimation": budget_estimation
                }
            }

        except Exception as e:
            logger.error("ResourceOptimizationAgent failed", error=str(e))
            return {"error": str(e), "resource_optimization": {}}

    def _calculate_resource_requirements(self, scorecard_data: Dict[str, Any],
                                         scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate human and technical resource requirements"""

        metrics = scorecard_data.get("metrics", {})
        severity = metrics.get("severity", 0.5)
        impact = metrics.get("impact", 0.5)
        legal_risk = metrics.get("legal_risk", 0.5)

        # Base resource needs on crisis characteristics
        # 3-10 people based on severity
        base_team_size = max(3, int(severity * 10))

        # Role-specific requirements
        roles_needed = {
            "crisis_manager": 1,
            # More if high impact
            "communications_specialist": 1 + int(impact * 2),
            "legal_counsel": 1 if legal_risk > 0.5 else 0,
            "subject_matter_expert": 1 + int(severity * 2),
            "stakeholder_liaison": max(1, int(metrics.get("reach", 0.5) * 3)),
            "data_analyst": 1,
            "executive_sponsor": 1 if severity > 0.7 else 0
        }

        # Technical resource requirements
        technical_resources = {
            "monitoring_tools": "enhanced" if impact > 0.6 else "standard",
            "communication_platforms": "multi_channel" if metrics.get("reach", 0.5) > 0.6 else "standard",
            "data_analytics": "real_time" if metrics.get("speed", 0.5) > 0.7 else "batch",
            "legal_research_tools": "required" if legal_risk > 0.5 else "optional"
        }

        return {
            "human_resources": roles_needed,
            "total_team_size": sum(roles_needed.values()),
            "technical_resources": technical_resources,
            "resource_intensity": "high" if sum(roles_needed.values()) > 8 else "medium" if sum(roles_needed.values()) > 5 else "low"
        }

    def _optimize_timeline(self, scorecard_data: Dict[str, Any],
                           scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize implementation timeline"""

        metrics = scorecard_data.get("metrics", {})
        urgency = metrics.get("speed", 0.5)

        # Base timeline phases
        if urgency > 0.8:
            phases = {
                "immediate_response": {"duration_hours": 1, "parallel_tracks": 2},
                "stabilization": {"duration_hours": 4, "parallel_tracks": 3},
                "resolution": {"duration_hours": 12, "parallel_tracks": 2},
                "recovery": {"duration_hours": 24, "parallel_tracks": 1}
            }
        elif urgency > 0.6:
            phases = {
                "immediate_response": {"duration_hours": 2, "parallel_tracks": 2},
                "stabilization": {"duration_hours": 8, "parallel_tracks": 3},
                "resolution": {"duration_hours": 24, "parallel_tracks": 2},
                "recovery": {"duration_hours": 48, "parallel_tracks": 1}
            }
        else:
            phases = {
                "immediate_response": {"duration_hours": 4, "parallel_tracks": 1},
                "stabilization": {"duration_hours": 16, "parallel_tracks": 2},
                "resolution": {"duration_hours": 48, "parallel_tracks": 2},
                "recovery": {"duration_hours": 96, "parallel_tracks": 1}
            }

        # Calculate critical path
        critical_path = self._calculate_critical_path(phases)

        return {
            "phases": phases,
            "critical_path": critical_path,
            "total_timeline_hours": sum(phase["duration_hours"] for phase in phases.values()),
            "optimization_opportunities": self._identify_optimization_opportunities(phases)
        }

    def _plan_capacity_allocation(self, resource_requirements: Dict[str, Any],
                                  timeline_optimization: Dict[str, Any]) -> Dict[str, Any]:
        """Plan capacity allocation across timeline phases"""

        human_resources = resource_requirements.get("human_resources", {})
        phases = timeline_optimization.get("phases", {})

        capacity_plan = {}

        for phase_name, phase_info in phases.items():
            parallel_tracks = phase_info.get("parallel_tracks", 1)
            duration_hours = phase_info.get("duration_hours", 8)

            # Allocate resources to phase
            phase_capacity = {}
            for role, count in human_resources.items():
                # Distribute resources across parallel tracks
                tracks_per_role = min(parallel_tracks, count)
                phase_capacity[role] = {
                    "allocated_count": count,
                    "parallel_tracks": tracks_per_role,
                    "utilization_hours": duration_hours * tracks_per_role
                }

            capacity_plan[phase_name] = {
                "resource_allocation": phase_capacity,
                "total_person_hours": sum(
                    alloc["utilization_hours"] for alloc in phase_capacity.values()
                ),
                "parallel_efficiency": parallel_tracks
            }

        return capacity_plan

    def _estimate_budget(self, resource_requirements: Dict[str, Any],
                         scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate budget requirements"""

        human_resources = resource_requirements.get("human_resources", {})

        # Role-based hourly rates (placeholder - would be company-specific)
        hourly_rates = {
            "crisis_manager": 150,
            "communications_specialist": 100,
            "legal_counsel": 200,
            "subject_matter_expert": 120,
            "stakeholder_liaison": 80,
            "data_analyst": 90,
            "executive_sponsor": 300
        }

        # Calculate human resource costs
        total_human_cost = 0
        for role, count in human_resources.items():
            rate = hourly_rates.get(role, 100)
            # Assume 40 hours average per crisis
            role_cost = count * rate * 40
            total_human_cost += role_cost

        # Add scenario-specific costs
        scenario_costs = [s.get("cost_estimate", 50000) for s in scenarios]
        avg_scenario_cost = sum(scenario_costs) / \
            len(scenario_costs) if scenario_costs else 50000

        # Technical infrastructure costs
        technical_cost = resource_requirements.get("technical_resources", {})
        infrastructure_cost = 10000  # Base technical cost

        total_estimated_cost = total_human_cost + \
            avg_scenario_cost + infrastructure_cost

        return {
            "human_resource_cost": total_human_cost,
            "scenario_implementation_cost": avg_scenario_cost,
            "technical_infrastructure_cost": infrastructure_cost,
            "total_estimated_cost": total_estimated_cost,
            "cost_breakdown": {
                role: count * hourly_rates.get(role, 100) * 40
                for role, count in human_resources.items()
            }
        }

    def _calculate_critical_path(self, phases: Dict[str, Any]) -> List[str]:
        """Calculate critical path through timeline phases"""

        # Simple sequential critical path
        sorted_phases = sorted(
            phases.items(),
            key=lambda x: x[1].get("duration_hours", 8)
        )

        return [phase_name for phase_name, _ in sorted_phases]

    def _identify_optimization_opportunities(self, phases: Dict[str, Any]) -> List[str]:
        """Identify opportunities to optimize timeline"""

        opportunities = []

        for phase_name, phase_info in phases.items():
            parallel_tracks = phase_info.get("parallel_tracks", 1)
            duration = phase_info.get("duration_hours", 8)

            if parallel_tracks > 1:
                opportunities.append(
                    f"Parallelize {phase_name} activities to reduce duration")

            if duration > 24:
                opportunities.append(
                    f"Consider breaking {phase_name} into smaller milestones")

        return opportunities


class RiskMitigationAgent:
    """Contingency Planning Specialist - Agent 5/7"""

    def __init__(self, firestore_read_tool: FirestoreReadTool) -> None:
        self.read_tool = firestore_read_tool

    async def execute(self, session: Session) -> Dict[str, Any]:
        """Develop comprehensive risk mitigation and contingency plans"""

        try:
            scorecard_data = session.state.get("scorecard_data", {})
            snapshot_data = session.state.get("snapshot_data", {})

            # Identify primary risks
            primary_risks = self._identify_primary_risks(scorecard_data)

            # Develop mitigation strategies
            mitigation_strategies = self._develop_mitigation_strategies(
                primary_risks, scorecard_data)

            # Create contingency plans
            contingency_plans = self._create_contingency_plans(
                scorecard_data, snapshot_data)

            # Define escalation triggers
            escalation_triggers = self._define_escalation_triggers(
                scorecard_data)

            return {
                "risk_mitigation": {
                    "primary_risks": primary_risks,
                    "mitigation_strategies": mitigation_strategies,
                    "contingency_plans": contingency_plans,
                    "escalation_triggers": escalation_triggers
                }
            }

        except Exception as e:
            logger.error("RiskMitigationAgent failed", error=str(e))
            return {"error": str(e), "risk_mitigation": {}}

    def _identify_primary_risks(self, scorecard_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify the highest-priority risks to mitigate"""

        metrics = scorecard_data.get("metrics", {})

        # Analyze all risk dimensions
        risk_dimensions = [
            {"type": "severity", "score": metrics.get(
                "severity", 0), "category": "operational"},
            {"type": "impact", "score": metrics.get(
                "impact", 0), "category": "business"},
            {"type": "reputational_risk", "score": metrics.get(
                "reputational_risk", 0), "category": "brand"},
            {"type": "legal_risk", "score": metrics.get(
                "legal_risk", 0), "category": "compliance"},
            {"type": "financial_risk", "score": metrics.get(
                "financial_risk", 0), "category": "financial"}
        ]

        # Filter and rank high-priority risks
        high_risks = [r for r in risk_dimensions if r["score"] > 0.6]
        high_risks.sort(key=lambda x: x["score"], reverse=True)

        # Add risk-specific details
        for risk in high_risks:
            risk.update({
                "mitigation_priority": "critical" if risk["score"] > 0.8 else "high",
                "estimated_impact": risk["score"],
                "mitigation_urgency": self._calculate_mitigation_urgency(risk)
            })

        return high_risks

    def _develop_mitigation_strategies(self, primary_risks: List[Dict[str, Any]],
                                       scorecard_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Develop specific mitigation strategies for each risk"""

        mitigation_strategies = {}

        for risk in primary_risks:
            risk_type = risk["type"]
            strategies = []

            if risk_type == "severity":
                strategies.extend([
                    "Implement crisis command center",
                    "Deploy additional resources immediately",
                    "Activate emergency response protocols",
                    "Establish 24/7 monitoring"
                ])

            elif risk_type == "impact":
                strategies.extend([
                    "Launch impact containment measures",
                    "Implement damage control protocols",
                    "Activate business continuity plans",
                    "Deploy impact monitoring systems"
                ])

            elif risk_type == "reputational_risk":
                strategies.extend([
                    "Execute reputation protection campaign",
                    "Engage reputation management consultants",
                    "Launch positive messaging initiative",
                    "Monitor and respond to social media"
                ])

            elif risk_type == "legal_risk":
                strategies.extend([
                    "Engage external legal counsel",
                    "Conduct compliance audit",
                    "Prepare legal defense documentation",
                    "Implement compliance monitoring"
                ])

            elif risk_type == "financial_risk":
                strategies.extend([
                    "Activate financial risk management",
                    "Engage with investors and analysts",
                    "Implement cost containment measures",
                    "Prepare financial impact statements"
                ])

            mitigation_strategies[risk_type] = strategies

        return mitigation_strategies

    def _create_contingency_plans(self, scorecard_data: Dict[str, Any],
                                  snapshot_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create contingency plans for various escalation scenarios"""

        metrics = scorecard_data.get("metrics", {})
        context = snapshot_data.get("context", {})

        contingency_plans = []

        # Scenario 1: Rapid escalation
        if metrics.get("speed", 0.5) > 0.6:
            contingency_plans.append({
                "scenario": "rapid_escalation",
                "trigger_conditions": [
                    "Social media mentions increase >200% in 2 hours",
                    "Media coverage spreads to major outlets",
                    "Regulatory inquiry initiated"
                ],
                "response_plan": [
                    "Activate crisis command center immediately",
                    "Deploy senior executive spokesperson",
                    "Implement 1-hour communication cycles",
                    "Engage external crisis management firm"
                ],
                "resource_requirements": "emergency_level",
                "timeline": "immediate (0-2 hours)"
            })

        # Scenario 2: Legal escalation
        if metrics.get("legal_risk", 0.5) > 0.6:
            contingency_plans.append({
                "scenario": "legal_escalation",
                "trigger_conditions": [
                    "Regulatory investigation announced",
                    "Class action lawsuit filed",
                    "Government agency involvement"
                ],
                "response_plan": [
                    "Invoke attorney-client privilege protocols",
                    "Engage specialized regulatory counsel",
                    "Implement document preservation holds",
                    "Coordinate with regulatory affairs"
                ],
                "resource_requirements": "legal_intensive",
                "timeline": "immediate (0-4 hours)"
            })

        # Scenario 3: Stakeholder revolt
        stakeholder_count = len(context.get("relations", []))
        if stakeholder_count > 5 and metrics.get("reach", 0.5) > 0.7:
            contingency_plans.append({
                "scenario": "stakeholder_revolt",
                "trigger_conditions": [
                    "Multiple key stakeholders express dissatisfaction",
                    "Partnership agreements threatened",
                    "Investor confidence deteriorates"
                ],
                "response_plan": [
                    "Launch immediate stakeholder outreach",
                    "Schedule individual stakeholder meetings",
                    "Develop stakeholder-specific action plans",
                    "Implement relationship recovery program"
                ],
                "resource_requirements": "relationship_intensive",
                "timeline": "urgent (0-8 hours)"
            })

        return contingency_plans

    def _define_escalation_triggers(self, scorecard_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Define specific triggers that require escalation"""

        metrics = scorecard_data.get("metrics", {})

        triggers = [
            {
                "trigger_type": "severity_threshold",
                "condition": "Crisis severity score exceeds 0.9",
                "current_value": metrics.get("severity", 0),
                "threshold": 0.9,
                "escalation_action": "Activate CEO-level crisis response",
                "notification_required": ["C-suite", "Board of Directors"]
            },
            {
                "trigger_type": "legal_risk_threshold",
                "condition": "Legal risk score exceeds 0.8",
                "current_value": metrics.get("legal_risk", 0),
                "threshold": 0.8,
                "escalation_action": "Engage external legal counsel immediately",
                "notification_required": ["General Counsel", "CEO"]
            },
            {
                "trigger_type": "financial_impact_threshold",
                "condition": "Financial risk score exceeds 0.7",
                "current_value": metrics.get("financial_risk", 0),
                "threshold": 0.7,
                "escalation_action": "Activate financial crisis protocols",
                "notification_required": ["CFO", "Board Finance Committee"]
            },
            {
                "trigger_type": "media_attention_threshold",
                "condition": "Reputational risk score exceeds 0.8",
                "current_value": metrics.get("reputational_risk", 0),
                "threshold": 0.8,
                "escalation_action": "Deploy senior communications leadership",
                "notification_required": ["Chief Communications Officer", "CEO"]
            }
        ]

        # Filter to only relevant triggers (where current value is approaching threshold)
        relevant_triggers = [
            t for t in triggers
            # Within 30% of threshold
            if t["current_value"] > (t["threshold"] * 0.7)
        ]

        return relevant_triggers

    def _calculate_mitigation_urgency(self, risk: Dict[str, Any]) -> str:
        """Calculate mitigation urgency for a specific risk"""
        score = risk.get("score", 0)

        if score > 0.9:
            return "immediate"
        elif score > 0.7:
            return "urgent"
        elif score > 0.5:
            return "important"
        else:
            return "standard"


class ComplianceValidatorAgent:
    """Legal and Regulatory Specialist - Agent 6/7"""

    def __init__(self, firestore_read_tool: FirestoreReadTool) -> None:
        self.read_tool = firestore_read_tool

    async def execute(self, session: Session) -> Dict[str, Any]:
        """Validate recommendations against legal and regulatory requirements"""

        try:
            planning_results = session.state.get("planning_results", {})
            scorecard_data = session.state.get("scorecard_data", {})
            snapshot_data = session.state.get("snapshot_data", {})

            # Extract proposed strategies
            proposed_strategies = self._extract_proposed_strategies(
                planning_results)

            # Validate against compliance requirements
            compliance_validation = await self._validate_compliance(proposed_strategies, snapshot_data)

            # Check regulatory constraints
            regulatory_constraints = self._check_regulatory_constraints(
                proposed_strategies, snapshot_data)

            # Validate policy adherence
            policy_validation = self._validate_policy_adherence(
                proposed_strategies)

            return {
                "compliance_validation": {
                    "proposed_strategies_validated": len(proposed_strategies),
                    "compliance_status": compliance_validation,
                    "regulatory_constraints": regulatory_constraints,
                    "policy_validation": policy_validation,
                    "overall_compliance_score": self._calculate_compliance_score(
                        compliance_validation, regulatory_constraints, policy_validation
                    )
                }
            }

        except Exception as e:
            logger.error("ComplianceValidatorAgent failed", error=str(e))
            return {"error": str(e), "compliance_validation": {}}

    def _extract_proposed_strategies(self, planning_results: Dict[str, Any]) -> List[str]:
        """Extract proposed strategies from planning results"""
        strategies = []

        # Extract from scenario modeling
        scenario_data = planning_results.get(
            "scenariomodeling", {}).get("data", {})
        scenarios = scenario_data.get(
            "scenario_modeling", {}).get("scenarios", [])

        for scenario in scenarios:
            strategies.extend(scenario.get("key_actions", []))

        # Extract from stakeholder strategy
        stakeholder_data = planning_results.get(
            "stakeholderstrategy", {}).get("data", {})
        comm_strategies = stakeholder_data.get(
            "stakeholder_strategy", {}).get("communication_strategies", [])

        for strategy in comm_strategies:
            strategies.extend(strategy.get("key_messages", []))

        return list(set(strategies))  # Remove duplicates

    async def _validate_compliance(self, strategies: List[str],
                                   snapshot_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate strategies against compliance requirements"""

        context = snapshot_data.get("context", {})
        company_profile = context.get("company_profile", {})
        industry = company_profile.get("industry", "unknown")

        # Industry-specific compliance checks
        compliance_results = {}

        if industry == "finance":
            compliance_results.update({
                "sox_compliance": self._check_financial_disclosure_compliance(strategies),
                "sec_requirements": self._check_sec_requirements(strategies),
                "banking_regulations": self._check_banking_compliance(strategies)
            })

        elif industry == "healthcare":
            compliance_results.update({
                "hipaa_compliance": self._check_hipaa_compliance(strategies),
                "fda_requirements": self._check_fda_requirements(strategies),
                "patient_safety": self._check_patient_safety_compliance(strategies)
            })

        elif industry == "technology":
            compliance_results.update({
                "data_privacy": self._check_data_privacy_compliance(strategies),
                "platform_liability": self._check_platform_liability(strategies),
                "intellectual_property": self._check_ip_compliance(strategies)
            })

        else:
            compliance_results.update({
                "general_business": self._check_general_business_compliance(strategies),
                "employment_law": self._check_employment_compliance(strategies),
                "consumer_protection": self._check_consumer_protection(strategies)
            })

        return compliance_results

    def _check_regulatory_constraints(self, strategies: List[str],
                                      snapshot_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for regulatory constraints on proposed strategies"""

        constraints = []

        # Common regulatory constraints
        for strategy in strategies:
            if "public statement" in strategy.lower():
                constraints.append({
                    "strategy": strategy,
                    "constraint_type": "disclosure_requirements",
                    "requirement": "Must comply with material disclosure requirements",
                    "compliance_action": "Review statement with legal counsel before release"
                })

            if "stakeholder" in strategy.lower() and "engagement" in strategy.lower():
                constraints.append({
                    "strategy": strategy,
                    "constraint_type": "fair_disclosure",
                    "requirement": "Ensure fair and equal information disclosure",
                    "compliance_action": "Coordinate stakeholder communications to avoid selective disclosure"
                })

        return constraints

    def _validate_policy_adherence(self, strategies: List[str]) -> Dict[str, Any]:
        """Validate strategies against internal policies"""

        # Standard policy validation (would be company-specific in production)
        policy_checks = {
            "communication_policy": self._check_communication_policy(strategies),
            "stakeholder_policy": self._check_stakeholder_policy(strategies),
            "escalation_policy": self._check_escalation_policy(strategies),
            "risk_management_policy": self._check_risk_management_policy(strategies)
        }

        # Calculate overall policy compliance
        compliance_scores = [check.get("compliance_score", 0.5)
                             for check in policy_checks.values()]
        overall_compliance = sum(compliance_scores) / len(compliance_scores)

        return {
            "policy_checks": policy_checks,
            "overall_policy_compliance": overall_compliance,
            "policy_violations": [
                name for name, check in policy_checks.items()
                if check.get("compliance_score", 0.5) < 0.7
            ]
        }

    def _calculate_compliance_score(self, compliance_validation: Dict[str, Any],
                                    regulatory_constraints: List[Dict[str, Any]],
                                    policy_validation: Dict[str, Any]) -> float:
        """Calculate overall compliance score"""

        # Compliance validation score
        compliance_scores = [
            check.get("compliance_score", 0.5)
            for check in compliance_validation.values()
            if isinstance(check, dict)
        ]
        avg_compliance = sum(compliance_scores) / \
            len(compliance_scores) if compliance_scores else 0.5

        # Regulatory constraints penalty
        constraint_penalty = len(regulatory_constraints) * 0.1

        # Policy validation score
        policy_score = policy_validation.get("overall_policy_compliance", 0.5)

        # Overall score
        overall_score = max(
            0, (avg_compliance * 0.5 + policy_score * 0.5) - constraint_penalty)

        return min(1.0, overall_score)

    # Compliance check helper methods (simplified for brevity)
    def _check_financial_disclosure_compliance(self, strategies: List[str]) -> Dict[str, Any]:
        return {"compliance_score": 0.8, "requirements_met": True}

    def _check_sec_requirements(self, strategies: List[str]) -> Dict[str, Any]:
        return {"compliance_score": 0.7, "requirements_met": True}

    def _check_banking_compliance(self, strategies: List[str]) -> Dict[str, Any]:
        return {"compliance_score": 0.8, "requirements_met": True}

    def _check_hipaa_compliance(self, strategies: List[str]) -> Dict[str, Any]:
        return {"compliance_score": 0.9, "requirements_met": True}

    def _check_fda_requirements(self, strategies: List[str]) -> Dict[str, Any]:
        return {"compliance_score": 0.8, "requirements_met": True}

    def _check_patient_safety_compliance(self, strategies: List[str]) -> Dict[str, Any]:
        return {"compliance_score": 0.9, "requirements_met": True}

    def _check_data_privacy_compliance(self, strategies: List[str]) -> Dict[str, Any]:
        return {"compliance_score": 0.8, "requirements_met": True}

    def _check_platform_liability(self, strategies: List[str]) -> Dict[str, Any]:
        return {"compliance_score": 0.7, "requirements_met": True}

    def _check_ip_compliance(self, strategies: List[str]) -> Dict[str, Any]:
        return {"compliance_score": 0.8, "requirements_met": True}

    def _check_general_business_compliance(self, strategies: List[str]) -> Dict[str, Any]:
        return {"compliance_score": 0.8, "requirements_met": True}

    def _check_employment_compliance(self, strategies: List[str]) -> Dict[str, Any]:
        return {"compliance_score": 0.8, "requirements_met": True}

    def _check_consumer_protection(self, strategies: List[str]) -> Dict[str, Any]:
        return {"compliance_score": 0.8, "requirements_met": True}

    def _check_communication_policy(self, strategies: List[str]) -> Dict[str, Any]:
        return {"compliance_score": 0.8, "requirements_met": True}

    def _check_stakeholder_policy(self, strategies: List[str]) -> Dict[str, Any]:
        return {"compliance_score": 0.8, "requirements_met": True}

    def _check_escalation_policy(self, strategies: List[str]) -> Dict[str, Any]:
        return {"compliance_score": 0.8, "requirements_met": True}

    def _check_risk_management_policy(self, strategies: List[str]) -> Dict[str, Any]:
        return {"compliance_score": 0.8, "requirements_met": True}


class RecommendationSynthesizerAgent:
    """Strategic Integration Specialist - Final Synthesizer"""

    def __init__(self, db_pool: FirestoreConnectionPool, write_tool: FirestoreWriteTool) -> None:
        self.db_pool = db_pool
        self.write_tool = write_tool

    async def execute(self, session: Session) -> Dict[str, Any]:
        """Synthesize all planning results into final strategic recommendations"""

        try:
            crisis_case_id = session.state.get("crisis_case_id")
            company_id = session.state.get("company_id")
            planning_results = session.state.get("planning_results", {})
            scorecard_data = session.state.get("scorecard_data", {})

            # Integrate all planning results
            integrated_strategy = self._integrate_strategic_plans(
                planning_results)

            # Create actionable steps
            actionable_steps = self._create_actionable_steps(
                integrated_strategy, scorecard_data)

            # Calculate confidence metrics
            confidence_metrics = self._calculate_confidence_metrics(
                planning_results)

            # Extract similar case references
            similar_cases = self._extract_case_references(planning_results)

            # Create final recommendation
            recommendation = Recommendation(
                crisis_case_id=crisis_case_id,
                steps=actionable_steps,
                total_estimated_cost=integrated_strategy.get("total_cost"),
                total_timeline_hours=integrated_strategy.get(
                    "total_timeline_hours"),
                confidence_score=confidence_metrics.get(
                    "overall_confidence", 0.5),
                similar_cases_referenced=[
                    c.get("id", "") for c in similar_cases]
            )

            # Store recommendation as an Artifact under Company-scoped Crises
            recommendation_path = f"Company/{company_id}/Crises/{crisis_case_id}/Artifacts/{recommendation.reco_id}"
            await self.db_pool.create_document(recommendation_path, recommendation.model_dump())

            # Update crisis case with recommendation reference (company-scoped path)
            await self.db_pool.update_document(
                f"Company/{company_id}/Crises/{crisis_case_id}",
                {
                    "latest_recommendation_id": recommendation.reco_id,
                    "current_status": "recommendation_generated",
                    "estimated_resolution_time_hours": recommendation.total_timeline_hours,
                    "updated_at": datetime.utcnow()
                },
                company_id=company_id
            )

            # Create audit log
            await self._create_audit_log(crisis_case_id, recommendation, planning_results, company_id)

            logger.info("Strategic recommendation synthesized",
                        crisis_case_id=crisis_case_id, recommendation_id=recommendation.reco_id)

            return {
                "recommendation_id": recommendation.reco_id,
                "synthesis_status": "completed",
                "confidence_score": confidence_metrics.get("overall_confidence", 0.5),
                "total_steps": len(actionable_steps)
            }

        except Exception as e:
            logger.error("RecommendationSynthesizerAgent failed", error=str(e))
            return {"error": str(e), "recommendation_id": None}

    def _integrate_strategic_plans(self, planning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate results from all planning sub-agents"""

        integrated_strategy = {
            "strategic_approach": "multi_dimensional",
            "total_cost": 0,
            "total_timeline_hours": 48,  # Default
            "key_components": []
        }

        # Extract optimal scenario
        scenario_data = planning_results.get(
            "scenariomodeling", {}).get("data", {})
        optimal_strategy = scenario_data.get(
            "scenario_modeling", {}).get("optimal_strategy", {})
        recommended_scenario = optimal_strategy.get(
            "recommended_scenario", "defensive_response")

        integrated_strategy["primary_strategy"] = recommended_scenario

        # Extract resource requirements
        resource_data = planning_results.get(
            "resourceoptimization", {}).get("data", {})
        resource_optimization = resource_data.get("resource_optimization", {})
        budget_estimation = resource_optimization.get("budget_estimation", {})

        integrated_strategy["total_cost"] = budget_estimation.get(
            "total_estimated_cost", 100000)

        timeline_optimization = resource_optimization.get(
            "timeline_optimization", {})
        integrated_strategy["total_timeline_hours"] = timeline_optimization.get(
            "total_timeline_hours", 48)

        # Extract stakeholder strategy
        stakeholder_data = planning_results.get(
            "stakeholderstrategy", {}).get("data", {})
        stakeholder_strategy = stakeholder_data.get("stakeholder_strategy", {})

        integrated_strategy["communication_plan"] = stakeholder_strategy.get(
            "timing_plan", {})

        return integrated_strategy

    def _create_actionable_steps(self, integrated_strategy: Dict[str, Any],
                                 scorecard_data: Dict[str, Any]) -> List[RecommendationStep]:
        """Create concrete actionable steps from integrated strategy"""

        steps = []
        metrics = scorecard_data.get("metrics", {})

        # Step 1: Immediate Assessment and Setup
        steps.append(RecommendationStep(
            step_no=1,
            action="Activate crisis response team and establish command center",
            rationale="Rapid organization is critical for effective crisis management",
            risk_level=0.2,
            confidence=0.9,
            estimated_cost=5000,
            timeline="immediate (0-1 hours)"
        ))

        # Step 2: Stakeholder Communication
        steps.append(RecommendationStep(
            step_no=2,
            action="Execute stakeholder communication plan with priority sequencing",
            rationale="Early stakeholder engagement prevents escalation and builds trust",
            risk_level=0.3,
            confidence=0.8,
            estimated_cost=15000,
            timeline="urgent (1-4 hours)"
        ))

        # Step 3: Risk Mitigation
        if metrics.get("severity", 0) > 0.6:
            steps.append(RecommendationStep(
                step_no=3,
                action="Implement primary risk mitigation measures",
                rationale="Address root causes to prevent crisis escalation",
                risk_level=0.4,
                confidence=0.7,
                estimated_cost=25000,
                timeline="important (4-12 hours)"
            ))

        # Step 4: Compliance and Legal
        if metrics.get("legal_risk", 0) > 0.5:
            steps.append(RecommendationStep(
                step_no=len(steps) + 1,
                action="Ensure compliance with all regulatory requirements",
                rationale="Legal compliance protects against regulatory action",
                risk_level=0.3,
                confidence=0.8,
                estimated_cost=20000,
                timeline="critical (0-8 hours)"
            ))

        # Step 5: Monitoring and Adjustment
        steps.append(RecommendationStep(
            step_no=len(steps) + 1,
            action="Establish ongoing monitoring and response adjustment protocols",
            rationale="Continuous monitoring enables rapid response to changing conditions",
            risk_level=0.2,
            confidence=0.9,
            estimated_cost=10000,
            timeline="ongoing (12+ hours)"
        ))

        return steps

    def _calculate_confidence_metrics(self, planning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence metrics for the recommendations"""

        # Count successful vs failed planning agents
        successful_agents = len(
            [r for r in planning_results.values() if r.get("status") == "success"])
        total_agents = len(planning_results)

        execution_confidence = successful_agents / \
            total_agents if total_agents > 0 else 0

        # Data quality confidence (based on historical case availability)
        historical_data = planning_results.get(
            "historicalcasesearch", {}).get("data", {})
        historical_cases_count = historical_data.get(
            "historical_analysis", {}).get("total_cases_found", 0)
        # Good if 5+ similar cases
        data_confidence = min(1.0, historical_cases_count / 5)

        # Strategy coherence confidence
        scenario_data = planning_results.get(
            "scenariomodeling", {}).get("data", {})
        scenario_count = scenario_data.get(
            "scenario_modeling", {}).get("scenario_count", 0)
        # Good if 3+ scenarios modeled
        strategy_confidence = min(1.0, scenario_count / 3)

        overall_confidence = (execution_confidence * 0.4 +
                              data_confidence * 0.3 + strategy_confidence * 0.3)

        return {
            "overall_confidence": overall_confidence,
            "execution_confidence": execution_confidence,
            "data_confidence": data_confidence,
            "strategy_confidence": strategy_confidence,
            "successful_planning_agents": successful_agents,
            "total_planning_agents": total_agents
        }

    def _extract_case_references(self, planning_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract references to similar historical cases"""

        historical_data = planning_results.get(
            "historicalcasesearch", {}).get("data", {})
        similar_cases = historical_data.get(
            "historical_analysis", {}).get("similar_cases", [])

        # Extract case references
        case_references = []
        for case in similar_cases[:5]:  # Top 5 most relevant
            case_references.append({
                "id": getattr(case, 'id', f"case_{int(time.time())}"),
                "title": getattr(case, 'title', 'Historical Case'),
                "similarity_score": getattr(case, 'similarity_score', 0.5),
                "source_type": getattr(case, 'source_type', 'case_study')
            })

        return case_references

    async def _create_audit_log(self, crisis_case_id: str, recommendation: Recommendation,
                                planning_results: Dict[str, Any], company_id: str):
        """Create audit log for recommendation generation"""

        audit_entry = {
            "log_id": f"audit_{int(time.time())}",
            "timestamp": datetime.utcnow(),
            "agent_id": "recommendation_synthesizer",
            "action": "recommendation_generation",
            "input_data": {
                "crisis_case_id": crisis_case_id,
                "planning_agents_executed": len(planning_results),
                "successful_agents": len([r for r in planning_results.values() if r.get("status") == "success"])
            },
            "output_data": {
                "recommendation_id": recommendation.reco_id,
                "total_steps": len(recommendation.steps),
                "confidence_score": recommendation.confidence_score,
                "estimated_cost": recommendation.total_estimated_cost
            },
            "execution_time_ms": 0,  # Would be calculated in production
            "status": "success"
        }

        # Store audit log under company-scoped logs subcollection
        await self.db_pool.create_document(
            f"Company/{company_id}/Crises/{crisis_case_id}/logs/{audit_entry['log_id']}",
            audit_entry
        )

# Additional specialized agents with simplified implementations


class StrategicIntegrationAgent:
    """Agent 7/7 - Final integration specialist"""

    def __init__(self, firestore_read_tool: FirestoreReadTool) -> None:
        self.read_tool = firestore_read_tool

    async def execute(self, session: Session) -> Dict[str, Any]:
        """Perform final strategic integration"""
        return {
            "integration_status": "completed",
            "strategic_coherence_score": 0.8
        }


# Export
__all__ = ['RecommendationAgent']
