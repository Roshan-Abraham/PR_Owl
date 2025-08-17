"""
FastAPI Crisis Management System
Coherent company-scoped entrypoint that reads/writes under Company/{company_id}/Crises and uses Artifacts subcollection.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import asyncio
import structlog

from models.schemas import (
    CrisisCase,
    CrisisSimulationRequest,
    CrisisResponse,
    CrisisSnapshot,
)
from infrastructure.firestore_client import FirestoreConnectionPool
from agents.agent_orchestrator import AgentOrchestrator
from infrastructure.monitoring import setup_monitoring

logger = structlog.get_logger()
app = FastAPI(title="Crisis Management System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared resources
db_pool = FirestoreConnectionPool()
agent_orchestrator = AgentOrchestrator(db_pool)


@app.on_event("startup")
async def on_startup():
    logger.info("startup")
    await db_pool.initialize()
    await setup_monitoring()


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("shutdown")
    await db_pool.close()


@app.post("/companies/{company_id}/simulate", response_model=CrisisResponse)
async def create_crisis_simulation(
    company_id: str,
    request: CrisisSimulationRequest,
    background_tasks: BackgroundTasks,
) -> CrisisResponse:
    """Create a new crisis under Company/{company_id}/Crises/{crisis_id} and run agents in background."""
    try:
        crisis_case_id = f"crisis_{uuid.uuid4().hex}"
        session_id = f"{company_id}:{crisis_case_id}"

        # Build crisis case. company_id stored on parent path; keep field optional in model.
        crisis_case = CrisisCase(
            id=crisis_case_id,
            session_id=session_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            origin_point={
                "type": "simulation",
                "source": getattr(request, "template_id", None),
                "metadata": getattr(request, "simulation_params", {}) or {},
            },
            nature=getattr(request, "nature", "simulation"),
            current_status="created",
            summary=f"Simulation created from template {getattr(request, 'template_id', '')}",
        )

        # Persist crisis case under company scope
        await db_pool.create_document(f"Company/{company_id}/Crises/{crisis_case_id}", crisis_case.model_dump())

        # Run agents asynchronously in background
        background_tasks.add_task(
            agent_workflow, crisis_case_id, company_id, session_id)

        return CrisisResponse(
            crisis_case_id=crisis_case_id,
            status="created",
            message="Crisis created and agents started in background",
        )

    except Exception as e:
        logger.error("create_crisis_simulation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/companies/{company_id}/crisis/{crisis_case_id}", response_model=Optional[CrisisCase])
async def get_crisis_case(company_id: str, crisis_case_id: str) -> Optional[CrisisCase]:
    try:
        doc = await db_pool.get_document(f"Company/{company_id}/Crises/{crisis_case_id}", company_id=company_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Crisis not found")
        return CrisisCase(**doc)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_crisis_case_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/companies/{company_id}/crisis/{crisis_case_id}/snapshot", response_model=Optional[CrisisSnapshot])
async def get_latest_snapshot(company_id: str, crisis_case_id: str) -> Optional[CrisisSnapshot]:
    try:
        crisis_doc = await db_pool.get_document(f"Company/{company_id}/Crises/{crisis_case_id}", company_id=company_id)
        if not crisis_doc:
            raise HTTPException(status_code=404, detail="Crisis not found")

        snapshot_id = crisis_doc.get("snapshot_id")
        if not snapshot_id:
            return None

        snapshot = await db_pool.get_document(f"Company/{company_id}/Crises/{crisis_case_id}/Artifacts/{snapshot_id}", company_id=company_id)
        if not snapshot:
            return None
        return CrisisSnapshot(**snapshot)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_latest_snapshot_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/companies/{company_id}/crises")
async def list_crises(company_id: str, limit: int = Query(50, ge=1, le=200), status: Optional[str] = None) -> List[CrisisCase]:
    try:
        filters = []
        if status:
            filters.append(("current_status", "==", status))

        docs = await db_pool.query_collection(f"Company/{company_id}/Crises", filters=filters, limit=limit, company_id=company_id)
        return [CrisisCase(**d) for d in docs]
    except Exception as e:
        logger.error("list_crises_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


async def agent_workflow(crisis_case_id: str, company_id: str, session_id: str):
    """Runs the complete orchestrated agent workflow with comprehensive error handling."""
    correlation_id = f"{crisis_case_id}_{int(datetime.utcnow().timestamp())}"
    
    try:
        logger.info("agent_workflow_start", 
                    crisis_case_id=crisis_case_id, 
                    company_id=company_id,
                    correlation_id=correlation_id)

        # Update status to processing
        await db_pool.update_document(
            f"Company/{company_id}/Crises/{crisis_case_id}",
            {
                "current_status": "processing", 
                "processing_started_at": datetime.utcnow(),
                "correlation_id": correlation_id,
                "updated_at": datetime.utcnow()
            },
            company_id=company_id,
        )

        # Run the complete workflow with built-in error handling and recovery
        workflow_results = await agent_orchestrator.run_complete_workflow(
            crisis_case_id, company_id, session_id
        )
        
        # Log comprehensive workflow completion
        logger.info("agent_workflow_complete", 
                    crisis_case_id=crisis_case_id,
                    correlation_id=correlation_id,
                    workflow_status=workflow_results.get("workflow_status"),
                    agents_executed=len(workflow_results.get("agents_executed", [])),
                    errors_count=len(workflow_results.get("errors", [])))
        
        return workflow_results
        
    except Exception as e:
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "correlation_id": correlation_id,
            "failure_timestamp": datetime.utcnow().isoformat()
        }
        
        logger.error("agent_workflow_critical_error", 
                     error=str(e),
                     crisis_case_id=crisis_case_id,
                     correlation_id=correlation_id,
                     exc_info=True)
        
        # Attempt graceful status rollback with multiple retry attempts
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await db_pool.update_document(
                    f"Company/{company_id}/Crises/{crisis_case_id}",
                    {
                        "current_status": "processing_failed", 
                        "error_details": error_details,
                        "processing_failed_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    },
                    company_id=company_id,
                )
                logger.info("error_status_rollback_success", 
                           crisis_case_id=crisis_case_id,
                           correlation_id=correlation_id,
                           attempt=attempt + 1)
                break
                
            except Exception as rollback_error:
                logger.warning("error_status_rollback_attempt_failed", 
                              crisis_case_id=crisis_case_id,
                              correlation_id=correlation_id,
                              attempt=attempt + 1,
                              rollback_error=str(rollback_error))
                
                if attempt == max_retries - 1:
                    logger.error("error_status_rollback_final_failure", 
                                crisis_case_id=crisis_case_id,
                                correlation_id=correlation_id,
                                original_error=str(e),
                                rollback_error=str(rollback_error))
                else:
                    # Exponential backoff before retry
                    await asyncio.sleep(2 ** attempt)
        
        # Re-raise the original exception for upstream handling
        raise


@app.get("/health")
async def health() -> Dict[str, Any]:
    try:
        await db_pool.health_check()
        return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
