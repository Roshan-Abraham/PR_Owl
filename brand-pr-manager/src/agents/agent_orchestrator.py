"""
Agent Orchestrator - Coordinates execution of all crisis management agents
Implements the main workflow from Context Collection -> Classification -> Recommendations
"""

import asyncio
import time
from typing import Dict, Any, Optional
from datetime import datetime
import structlog

from .context_collector_agent import ContextCollectorAgent
from .classification_agent import ClassificationAgent
from .recommendation_agent import RecommendationAgent
from infrastructure.firestore_client import FirestoreConnectionPool
from tools.vector_search_tool import VectorSearchTool
from infrastructure.monitoring import metrics_collector, structured_logger

logger = structlog.get_logger()

class AgentOrchestrator:
    """
    Main orchestrator for the 3-agent crisis management workflow
    Coordinates Context Collector -> Classification -> Recommendation agents
    """
    
    def __init__(self, db_pool: FirestoreConnectionPool) -> None:
        self.db_pool = db_pool
        
        # Initialize vector search
        self.vector_search = VectorSearchTool()
        
        # Initialize agents
        self.context_collector = ContextCollectorAgent(db_pool, self.vector_search)
        self.classification_agent = ClassificationAgent(db_pool)
        self.recommendation_agent = RecommendationAgent(db_pool, self.vector_search)
        
        # Orchestrator metrics
        self.execution_stats = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "avg_workflow_time_ms": 0.0
        }
        
    async def run_complete_workflow(self, crisis_case_id: str, company_id: str, session_id: str) -> Dict[str, Any]:
        """
        Run the complete 3-agent workflow for crisis management with comprehensive tracing
        Returns comprehensive results from all agents
        """
        start_time = time.time()
        self.execution_stats["total_workflows"] += 1
        
        # Create unique trace ID for this workflow
        trace_id = f"workflow_{crisis_case_id}_{int(start_time)}"
        
        workflow_results = {
            "trace_id": trace_id,
            "crisis_case_id": crisis_case_id,
            "company_id": company_id,
            "session_id": session_id,
            "workflow_start": datetime.utcnow().isoformat(),
            "agents_executed": [],
            "errors": []
        }
        
        # Start workflow trace
        structured_logger.start_trace(
            trace_id, 
            "complete_crisis_workflow",
            crisis_case_id=crisis_case_id,
            company_id=company_id,
            session_id=session_id
        )
        
        try:
            logger.info("Starting complete crisis management workflow", 
                       crisis_case_id=crisis_case_id, company_id=company_id, trace_id=trace_id)
            
            # Phase 1: Context Collection
            phase_start = time.time()
            context_result = await self._run_context_collection_phase(
                crisis_case_id, company_id, session_id, workflow_results, trace_id
            )
            phase_duration = time.time() - phase_start
            structured_logger.log_workflow_phase(
                trace_id, "context_collection", "success" if context_result else "failed", 
                phase_duration, crisis_case_id, company_id, context_result
            )
            
            # Phase 2: Classification
            phase_start = time.time()
            classification_result = await self._run_classification_phase(
                crisis_case_id, company_id, session_id, workflow_results, context_result, trace_id
            )
            phase_duration = time.time() - phase_start
            structured_logger.log_workflow_phase(
                trace_id, "classification", "success" if classification_result else "failed", 
                phase_duration, crisis_case_id, company_id, classification_result
            )
            
            # Phase 3: Recommendations
            phase_start = time.time()
            recommendation_result = await self._run_recommendation_phase(
                crisis_case_id, company_id, session_id, workflow_results, classification_result, trace_id
            )
            phase_duration = time.time() - phase_start
            structured_logger.log_workflow_phase(
                trace_id, "recommendations", "success" if recommendation_result else "failed", 
                phase_duration, crisis_case_id, company_id, recommendation_result
            )
            
            # Finalize workflow
            workflow_results.update({
                "context_snapshot_id": context_result,
                "classification_scorecard_id": classification_result,
                "recommendation_id": recommendation_result,
                "workflow_status": "completed",
                "workflow_end": datetime.utcnow().isoformat()
            })
            
            # Update final crisis status
            await self._finalize_crisis_case(crisis_case_id, workflow_results)
            
            # Update metrics
            execution_time = (time.time() - start_time) * 1000
            self.execution_stats["successful_workflows"] += 1
            self.execution_stats["avg_workflow_time_ms"] = (
                (self.execution_stats["avg_workflow_time_ms"] * (self.execution_stats["total_workflows"] - 1) 
                 + execution_time) / self.execution_stats["total_workflows"]
            )
            
            # Record metrics
            metrics_collector.record_agent_execution(
                "complete_workflow", company_id, "success", execution_time / 1000
            )
            
            # End trace successfully
            trace_summary = structured_logger.end_trace(
                trace_id, "success", 
                total_execution_time_ms=execution_time,
                agents_executed_count=len(workflow_results["agents_executed"]),
                final_outputs={
                    "snapshot_id": context_result,
                    "scorecard_id": classification_result,
                    "recommendation_id": recommendation_result
                }
            )
            workflow_results["trace_summary"] = trace_summary
            
            logger.info("Complete crisis management workflow finished", 
                       crisis_case_id=crisis_case_id, 
                       trace_id=trace_id,
                       execution_time_ms=execution_time,
                       agents_executed=len(workflow_results["agents_executed"]))
            
            return workflow_results
            
        except Exception as e:
            self.execution_stats["failed_workflows"] += 1
            execution_time = (time.time() - start_time) * 1000
            
            workflow_results.update({
                "workflow_status": "failed",
                "workflow_error": str(e),
                "workflow_end": datetime.utcnow().isoformat()
            })
            
            # Record failure metrics
            metrics_collector.record_agent_execution(
                "complete_workflow", company_id, "failed", execution_time / 1000
            )
            
            # End trace with failure
            trace_summary = structured_logger.end_trace(
                trace_id, "failed", 
                total_execution_time_ms=execution_time,
                agents_executed_count=len(workflow_results["agents_executed"]),
                error_details=str(e),
                error_type=type(e).__name__
            )
            workflow_results["trace_summary"] = trace_summary
            
            # Log structured error
            structured_logger.log_agent_execution(
                "complete_workflow", crisis_case_id, company_id, "failed", 
                execution_time / 1000, str(e)
            )
            
            # Update crisis case with error status
            await self.db_pool.update_document(
                f"crises/{crisis_case_id}",
                {
                    "current_status": "error",
                    "error_details": str(e),
                    "updated_at": datetime.utcnow()
                }
            )
            
            logger.error("Complete crisis management workflow failed", 
                        crisis_case_id=crisis_case_id, trace_id=trace_id, error=str(e))
            
            return workflow_results
            
    async def _run_context_collection_phase(self, crisis_case_id: str, company_id: str, 
                                           session_id: str, workflow_results: Dict[str, Any], trace_id: str) -> Optional[str]:
        """Execute Phase 1: Context Collection"""
        try:
            logger.info("Starting context collection phase", 
                       crisis_case_id=crisis_case_id, trace_id=trace_id)
            
            agent_start = time.time()
            snapshot_id = await self.context_collector.collect_context(
                crisis_case_id, company_id, session_id
            )
            agent_duration = time.time() - agent_start
            
            # Log agent execution step
            structured_logger.log_agent_step(
                trace_id, "context_collector", "execute", "success", agent_duration,
                crisis_case_id, company_id,
                output_summary=f"Generated snapshot: {snapshot_id}"
            )
            
            workflow_results["agents_executed"].append({
                "agent": "context_collector",
                "status": "success",
                "output_id": snapshot_id,
                "duration_seconds": agent_duration,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.info("Context collection phase completed", 
                       crisis_case_id=crisis_case_id, trace_id=trace_id, 
                       snapshot_id=snapshot_id, duration_seconds=agent_duration)
            
            return snapshot_id
            
        except Exception as e:
            error_msg = f"Context collection failed: {str(e)}"
            agent_duration = time.time() - agent_start if 'agent_start' in locals() else 0
            
            # Log failed agent step
            structured_logger.log_agent_step(
                trace_id, "context_collector", "execute", "failed", agent_duration,
                crisis_case_id, company_id,
                error_details=error_msg
            )
            
            workflow_results["errors"].append({
                "phase": "context_collection",
                "error": error_msg,
                "duration_seconds": agent_duration,
                "timestamp": datetime.utcnow().isoformat()
            })
            workflow_results["agents_executed"].append({
                "agent": "context_collector", 
                "status": "failed",
                "error": error_msg,
                "duration_seconds": agent_duration,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.error("Context collection phase failed", 
                        crisis_case_id=crisis_case_id, trace_id=trace_id, error=str(e))
            raise
            
    async def _run_classification_phase(self, crisis_case_id: str, company_id: str,
                                      session_id: str, workflow_results: Dict[str, Any],
                                      snapshot_id: Optional[str], trace_id: str) -> Optional[str]:
        """Execute Phase 2: Classification with full ClassificationAgent"""
        try:
            logger.info("Starting classification phase", crisis_case_id=crisis_case_id, trace_id=trace_id)
            agent_start = time.time()
            
            if not snapshot_id:
                raise ValueError("No snapshot available for classification")
            
            # Run classification agent
            scorecard_id = await self.classification_agent.classify_crisis(
                snapshot_id, crisis_case_id, company_id, session_id
            )
            agent_duration = time.time() - agent_start
            
            # Log agent execution step
            structured_logger.log_agent_step(
                trace_id, "classification_agent", "execute", "success", agent_duration,
                crisis_case_id, company_id,
                input_summary=f"Using snapshot: {snapshot_id}",
                output_summary=f"Generated scorecard: {scorecard_id}"
            )
            
            workflow_results["agents_executed"].append({
                "agent": "classification_agent",
                "status": "success",
                "output_id": scorecard_id,
                "duration_seconds": agent_duration,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.info("Classification phase completed", 
                       crisis_case_id=crisis_case_id, trace_id=trace_id,
                       scorecard_id=scorecard_id, duration_seconds=agent_duration)
            
            return scorecard_id
            
        except Exception as e:
            error_msg = f"Classification failed: {str(e)}"
            agent_duration = time.time() - agent_start if 'agent_start' in locals() else 0
            
            # Log failed agent step
            structured_logger.log_agent_step(
                trace_id, "classification_agent", "execute", "failed", agent_duration,
                crisis_case_id, company_id,
                input_summary=f"Using snapshot: {snapshot_id}",
                error_details=error_msg
            )
            
            workflow_results["errors"].append({
                "phase": "classification",
                "error": error_msg,
                "duration_seconds": agent_duration,
                "timestamp": datetime.utcnow().isoformat()
            })
            workflow_results["agents_executed"].append({
                "agent": "classification_agent", 
                "status": "failed",
                "error": error_msg,
                "duration_seconds": agent_duration,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.error("Classification phase failed", 
                        crisis_case_id=crisis_case_id, trace_id=trace_id, error=str(e))
            return None
            
    async def _run_recommendation_phase(self, crisis_case_id: str, company_id: str,
                                      session_id: str, workflow_results: Dict[str, Any],
                                      scorecard_id: Optional[str], trace_id: str) -> Optional[str]:
        """Execute Phase 3: Recommendations with full RecommendationAgent"""
        try:
            logger.info("Starting recommendation phase", crisis_case_id=crisis_case_id, trace_id=trace_id)
            agent_start = time.time()
            
            if not scorecard_id:
                raise ValueError("No scorecard available for recommendations")
                
            # Get snapshot_id from crisis case
            crisis_case = await self.db_pool.get_document(f"crises/{crisis_case_id}", company_id)
            snapshot_id = crisis_case.get("snapshot_id") if crisis_case else None
            
            if not snapshot_id:
                raise ValueError("No snapshot available for recommendations")
            
            # Run recommendation agent
            recommendation_id = await self.recommendation_agent.generate_recommendations(
                crisis_case_id, scorecard_id, snapshot_id, company_id, session_id
            )
            agent_duration = time.time() - agent_start
            
            # Log agent execution step
            structured_logger.log_agent_step(
                trace_id, "recommendation_agent", "execute", "success", agent_duration,
                crisis_case_id, company_id,
                input_summary=f"Using scorecard: {scorecard_id}, snapshot: {snapshot_id}",
                output_summary=f"Generated recommendations: {recommendation_id}"
            )
            
            workflow_results["agents_executed"].append({
                "agent": "recommendation_agent",
                "status": "success",
                "output_id": recommendation_id,
                "duration_seconds": agent_duration,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.info("Recommendation phase completed", 
                       crisis_case_id=crisis_case_id, trace_id=trace_id,
                       recommendation_id=recommendation_id, duration_seconds=agent_duration)
            
            return recommendation_id
            
        except Exception as e:
            error_msg = f"Recommendation failed: {str(e)}"
            agent_duration = time.time() - agent_start if 'agent_start' in locals() else 0
            
            # Log failed agent step
            structured_logger.log_agent_step(
                trace_id, "recommendation_agent", "execute", "failed", agent_duration,
                crisis_case_id, company_id,
                input_summary=f"Using scorecard: {scorecard_id}, snapshot: {snapshot_id}",
                error_details=error_msg
            )
            
            workflow_results["errors"].append({
                "phase": "recommendation",
                "error": error_msg,
                "duration_seconds": agent_duration,
                "timestamp": datetime.utcnow().isoformat()
            })
            workflow_results["agents_executed"].append({
                "agent": "recommendation_agent", 
                "status": "failed",
                "error": error_msg,
                "duration_seconds": agent_duration,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.error("Recommendation phase failed", 
                        crisis_case_id=crisis_case_id, trace_id=trace_id, error=str(e))
            return None
            
    async def _finalize_crisis_case(self, crisis_case_id: str, workflow_results: Dict[str, Any]) -> None:
        """Finalize crisis case with workflow results"""
        
        # Determine final status
        if workflow_results.get("workflow_status") == "completed":
            final_status = "recommendation_generated"
        elif any(agent.get("status") == "success" for agent in workflow_results.get("agents_executed", [])):
            final_status = "partially_processed"
        else:
            final_status = "processing_failed"
            
        # Update crisis case
        await self.db_pool.update_document(
            f"crises/{crisis_case_id}",
            {
                "current_status": final_status,
                "workflow_completed_at": datetime.utcnow(),
                "agents_executed_count": len(workflow_results.get("agents_executed", [])),
                "errors_encountered": len(workflow_results.get("errors", [])),
                "updated_at": datetime.utcnow()
            }
        )
        
        # Update dashboard if enabled
        if workflow_results.get("company_id"):
            try:
                await self._update_company_dashboard(
                    workflow_results["company_id"], final_status
                )
            except Exception as e:
                logger.warning("Dashboard update failed", 
                              company_id=workflow_results["company_id"], error=str(e))
                
    async def _update_company_dashboard(self, company_id: str, status: str) -> None:
        """Update company dashboard metrics"""
        
        # Determine counter updates based on status
        counter_updates = {}
        
        if status == "recommendation_generated":
            counter_updates["num_resolved_24h"] = 1
        elif status in ["partially_processed", "processing_failed"]:
            counter_updates["num_active"] = 1
            
        if status == "recommendation_generated":
            counter_updates["num_active"] = -1  # Remove from active
            
        # Apply counter updates if any
        if counter_updates:
            try:
                from tools.mcp_tools import FirestoreWriteTool
                write_tool = FirestoreWriteTool(self.db_pool)
                
                await write_tool.update_counters(
                    f"dashboards/{company_id}",
                    counter_updates,
                    company_id
                )
                
                logger.debug("Dashboard updated", 
                           company_id=company_id, updates=counter_updates)
                           
            except Exception as e:
                logger.warning("Dashboard counter update failed", 
                              company_id=company_id, error=str(e))
                
    # Individual agent execution methods (for manual triggers)
    
    async def run_context_collector(self, crisis_case_id: str, company_id: str, session_id: str) -> str:
        """Run only the context collector agent"""
        return await self.context_collector.collect_context(crisis_case_id, company_id, session_id)
        
    async def run_classification_agent(self, crisis_case_id: str, company_id: str, session_id: str) -> Optional[str]:
        """Run only the classification agent"""
        # Get the latest snapshot for classification
        crisis_case = await self.db_pool.get_document(f"crises/{crisis_case_id}", company_id)
        snapshot_id = crisis_case.get("snapshot_id") if crisis_case else None
        
        if not snapshot_id:
            raise ValueError("No snapshot available for classification")
            
        return await self.classification_agent.classify_crisis(snapshot_id, crisis_case_id, company_id, session_id)
        
    async def run_recommendation_agent(self, crisis_case_id: str, company_id: str, session_id: str) -> Optional[str]:
        """Run only the recommendation agent"""
        # Get the latest scorecard and snapshot for recommendations
        crisis_case = await self.db_pool.get_document(f"crises/{crisis_case_id}", company_id)
        if not crisis_case:
            raise ValueError("Crisis case not found")
            
        scorecard_id = crisis_case.get("latest_scorecard_id")
        snapshot_id = crisis_case.get("snapshot_id")
        
        if not scorecard_id:
            raise ValueError("No scorecard available for recommendations")
        if not snapshot_id:
            raise ValueError("No snapshot available for recommendations")
            
        return await self.recommendation_agent.generate_recommendations(
            crisis_case_id, scorecard_id, snapshot_id, company_id, session_id
        )
        
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get orchestrator execution statistics"""
        return {
            **self.execution_stats,
            "success_rate": (
                self.execution_stats["successful_workflows"] / 
                max(1, self.execution_stats["total_workflows"])
            ) * 100,
            "context_collector_stats": self.context_collector.execution_metrics
        }

# Export
__all__ = ['AgentOrchestrator']