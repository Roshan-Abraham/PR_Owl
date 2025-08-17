"""
Comprehensive data models and schemas for the Crisis Management System
Implements all Firestore collections and document structures from the PRP
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import uuid


# Enums for constrained fields
class CrisisNature(str, Enum):
    PRODUCT_DEFECT = "product_defect"
    REGULATORY = "regulatory"
    SOCIAL = "social"
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    LEGAL = "legal"


class CrisisStatus(str, Enum):
    CREATED = "created"
    CONTEXT_COLLECTED = "context_collected"
    CLASSIFIED = "classified"
    RECOMMENDATION_GENERATED = "recommendation_generated"
    ACTION_PLANNED = "action_planned"
    RESOLVED = "resolved"
    ARCHIVED = "archived"


class StakeholderType(str, Enum):
    CUSTOMER = "customer"
    PARTNER = "partner"
    INVESTOR = "investor"
    REGULATOR = "regulator"
    MEDIA = "media"


class OriginPointType(str, Enum):
    SIMULATION = "simulation"
    REAL = "real"
    EXTERNAL = "external"

# Core Data Models


class OriginPoint(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    type: OriginPointType
    source: str  # template_id, news_url, social_post_id, etc.
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContactInfo(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    name: str
    role: str
    email: str
    phone: Optional[str] = None


class Stakeholder(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    name: str
    influence_score: float = Field(ge=0.0, le=1.0)
    contact_method: str


class CompanySettings(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    notification_preferences: Dict[str, bool] = Field(default_factory=dict)
    escalation_thresholds: Dict[str, float] = Field(default_factory=dict)
    ai_model_preferences: Dict[str, str] = Field(default_factory=dict)


class CompanyMetadata(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    subscription_tier: str = "standard"
    feature_flags: Dict[str, bool] = Field(default_factory=dict)


class TrendData(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    severity_trend_7d: List[float] = Field(default_factory=list)
    volume_trend_7d: List[int] = Field(default_factory=list)


class DashboardSummary(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    num_crises_total: int = 0
    num_active: int = 0
    num_critical: int = 0
    num_resolved_24h: int = 0
    avg_resolution_time_hours: float = 0.0


class CompanyProfile(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    id: str
    name: str
    timezone: str = "UTC"
    industry: str
    brand_voice: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    settings: CompanySettings = Field(default_factory=CompanySettings)
    metadata: CompanyMetadata = Field(default_factory=CompanyMetadata)


class CompanyDetails(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    mission: str
    values: List[str] = Field(default_factory=list)
    ethics: List[str] = Field(default_factory=list)
    bio: str
    contacts: List[ContactInfo] = Field(default_factory=list)
    key_stakeholders: List[Stakeholder] = Field(default_factory=list)


class CompanyEvent(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    tags: List[str] = Field(default_factory=list)
    demographics: Dict[str, Any] = Field(default_factory=dict)
    start_time: datetime
    end_time: Optional[datetime] = None
    impact_estimate: float = Field(ge=0.0, le=1.0)
    stakeholder_involvement: List[str] = Field(default_factory=list)
    media_coverage_expected: bool = False


class StakeholderRelation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    relation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: StakeholderType
    importance_score: float = Field(ge=0.0, le=1.0)
    contact_info: Optional[ContactInfo] = None
    communication_history: List[Dict[str, Any]] = Field(default_factory=list)
    sentiment_history: List[Dict[str, float]] = Field(default_factory=list)


class SimulationTemplate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    template_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    scenario_type: str
    complexity_level: str
    learning_objectives: List[str] = Field(default_factory=list)
    default_parameters: Dict[str, Any] = Field(default_factory=dict)
    estimated_duration: int  # minutes


class KnowledgeBaseEntry(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    kb_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    category: str
    lessons_learned: str
    outcome_summary: str
    vector_ids: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

# Crisis-related Models


class CrisisCase(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    id: str
    # company_id is derived from the parent Company document path (Company/{company_id}/Crises/{id})
    # and should not be required on nested documents.
    company_id: Optional[str] = None
    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    origin_point: OriginPoint
    nature: CrisisNature = CrisisNature.SOCIAL
    current_status: CrisisStatus = CrisisStatus.CREATED
    primary_class: Optional[str] = None
    severity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    affected_stakeholders: List[str] = Field(default_factory=list)
    estimated_resolution_time_hours: Optional[int] = None
    snapshot_id: Optional[str] = None
    latest_scorecard_id: Optional[str] = None
    latest_recommendation_id: Optional[str] = None
    summary: str = ""
    error_details: Optional[str] = None


class CrisisContext(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    company_profile: Dict[str, Any] = Field(default_factory=dict)
    recent_events: List[Dict[str, Any]] = Field(default_factory=list)
    relations: List[Dict[str, Any]] = Field(default_factory=list)
    social_context: Dict[str, Any] = Field(default_factory=dict)
    last_24h_activity: Dict[str, Any] = Field(default_factory=dict)


class CrisisSnapshot(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    snapshot_id: str = Field(
        default_factory=lambda: f"snapshot_{int(datetime.utcnow().timestamp())}_{uuid.uuid4().hex[:8]}")
    crisis_case_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    context: CrisisContext
    agent_session_id: str
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)


class ScorecardMetrics(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    severity: float = Field(ge=0.0, le=1.0)
    impact: float = Field(ge=0.0, le=1.0)
    speed: float = Field(ge=0.0, le=1.0)
    reach: float = Field(ge=0.0, le=1.0)
    reputational_risk: float = Field(ge=0.0, le=1.0)
    legal_risk: float = Field(ge=0.0, le=1.0)
    financial_risk: float = Field(ge=0.0, le=1.0)


class AffectedEntity(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    entity_id: str
    relation_score: float = Field(ge=0.0, le=1.0)
    exposure: float = Field(ge=0.0, le=1.0)


class Scorecard(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    scorecard_id: str = Field(
        default_factory=lambda: f"scorecard_{int(datetime.utcnow().timestamp())}")
    crisis_case_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metrics: ScorecardMetrics
    affected_entities: List[AffectedEntity] = Field(default_factory=list)
    sub_agent_results: Dict[str, Any] = Field(default_factory=dict)
    confidence_metrics: Dict[str, float] = Field(default_factory=dict)


class RecommendationStep(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    step_no: int
    action: str
    rationale: str
    risk_level: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    estimated_cost: Optional[float] = None
    timeline: str  # e.g., "immediate", "2-4 hours", "1-2 days"


class Recommendation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    reco_id: str = Field(
        default_factory=lambda: f"reco_{int(datetime.utcnow().timestamp())}")
    crisis_case_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    steps: List[RecommendationStep] = Field(default_factory=list)
    total_estimated_cost: Optional[float] = None
    total_timeline_hours: Optional[int] = None
    confidence_score: float = Field(ge=0.0, le=1.0)
    similar_cases_referenced: List[str] = Field(default_factory=list)


class AgentLog(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    log_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_id: str
    sub_agent_id: Optional[str] = None
    action: str
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: int
    status: str  # "success", "error", "timeout"
    error_details: Optional[str] = None


class AgentSession(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    agent_coordination_state: Dict[str, Any] = Field(default_factory=dict)
    memory_bank_references: List[str] = Field(default_factory=list)
    current_step: str
    completed_steps: List[str] = Field(default_factory=list)

# Global Agent Tracking


class PerformanceMetrics(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    execution_time_ms: int
    memory_used: Optional[int] = None
    tokens_consumed: Optional[int] = None


class AgentRun(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    # company_id is optional because AgentRuns are stored under Company/{company_id}/AgentRuns
    company_id: Optional[str] = None
    crisis_case_id: str
    agent_type: str
    sub_agents_involved: List[str] = Field(default_factory=list)
    start_timestamp: datetime = Field(default_factory=datetime.utcnow)
    end_timestamp: Optional[datetime] = None
    status: str  # "running", "completed", "failed"
    error_details: Optional[str] = None
    performance_metrics: PerformanceMetrics

# Vector Database Models


class VectorMetadata(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    object_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    milvus_collection: str
    vector_id: str
    source_type: str  # "case_study", "company_knowledge", "external"
    source_id: str
    # Vector metadata may be global or company-scoped; keep as optional
    company_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    embeddings_model: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

# Dashboard Models


class Dashboard(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    company_id: str
    summary: DashboardSummary
    trend_data: TrendData
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    alert_thresholds: Dict[str, float] = Field(default_factory=dict)

# Request/Response Models for API


class CrisisSimulationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    company_id: str
    template_id: Optional[str] = None
    nature: Optional[CrisisNature] = None
    simulation_params: Optional[Dict[str, Any]] = None


class CrisisResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    crisis_case_id: str
    status: str
    message: str

# Search and Vector Models


class SearchResult(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    id: str
    similarity_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_id: str
    source_type: str
    title: str
    summary: str


class QueryFilter(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    field: str
    operator: str  # "==", "!=", ">=", "<=", ">", "<", "in", "array-contains"
    value: Any


class QueryOptions(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    limit: Optional[int] = None
    offset: Optional[int] = None
    order_by: Optional[str] = None
    order_direction: str = "asc"

# Validation helpers


# Field validators for score validation
class ScoreValidation:
    @field_validator('severity_score', 'confidence_score', mode='before')
    @classmethod
    def validate_score_range(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError('Score must be between 0.0 and 1.0')
        return v


# Export all models for easy importing
__all__ = [
    'CrisisCase', 'CrisisSnapshot', 'Scorecard', 'Recommendation',
    'CompanyProfile', 'CompanyDetails', 'CompanyEvent', 'StakeholderRelation',
    'SimulationTemplate', 'KnowledgeBaseEntry', 'AgentLog', 'AgentSession',
    'AgentRun', 'VectorMetadata', 'Dashboard', 'CrisisSimulationRequest',
    'CrisisResponse', 'SearchResult', 'QueryFilter', 'QueryOptions',
    'CrisisNature', 'CrisisStatus', 'StakeholderType', 'OriginPointType'
]
