"""
Monitoring, Logging, and Observability Infrastructure
Implements comprehensive monitoring for the Crisis Management System
"""

import time
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import structlog
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import logging.config

# OpenTelemetry imports (if available)
try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.gcp.monitoring import CloudMonitoringMetricsExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.trace import TracerProvider
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

from .config import settings, get_monitoring_config, get_logging_config

# Global logger
logger = structlog.get_logger()

# Prometheus metrics registry
metrics_registry = CollectorRegistry()

# Core application metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code'],
    registry=metrics_registry
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint'],
    registry=metrics_registry
)

# Agent execution metrics
agent_executions_total = Counter(
    'agent_executions_total',
    'Total agent executions',
    ['agent_type', 'company_id', 'status'],
    registry=metrics_registry
)

agent_execution_duration_seconds = Histogram(
    'agent_execution_duration_seconds',
    'Agent execution duration',
    ['agent_type'],
    registry=metrics_registry
)

# Database metrics
database_operations_total = Counter(
    'database_operations_total',
    'Total database operations',
    ['operation_type', 'collection', 'status'],
    registry=metrics_registry
)

database_operation_duration_seconds = Histogram(
    'database_operation_duration_seconds',
    'Database operation duration',
    ['operation_type', 'collection'],
    registry=metrics_registry
)

# Cache metrics
cache_operations_total = Counter(
    'cache_operations_total',
    'Total cache operations',
    ['operation_type', 'result'],
    registry=metrics_registry
)

# Vector search metrics
vector_searches_total = Counter(
    'vector_searches_total', 
    'Total vector searches',
    ['company_id', 'search_type', 'status'],
    registry=metrics_registry
)

vector_search_duration_seconds = Histogram(
    'vector_search_duration_seconds',
    'Vector search duration',
    ['search_type'],
    registry=metrics_registry
)

# System health metrics
active_connections_gauge = Gauge(
    'active_database_connections',
    'Active database connections',
    registry=metrics_registry
)

memory_usage_bytes = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes',
    registry=metrics_registry
)

crisis_cases_gauge = Gauge(
    'active_crisis_cases_total',
    'Total active crisis cases',
    ['company_id'],
    registry=metrics_registry
)

class MetricsCollector:
    """Centralized metrics collection and reporting"""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.agent_executions = {}
        self.performance_stats = {}
        
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        http_requests_total.labels(
            method=method,
            endpoint=endpoint, 
            status_code=str(status_code)
        ).inc()
        
        http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
        self.request_count += 1
        
    def record_agent_execution(self, agent_type: str, company_id: str, 
                             status: str, duration: float):
        """Record agent execution metrics"""
        agent_executions_total.labels(
            agent_type=agent_type,
            company_id=company_id,
            status=status
        ).inc()
        
        agent_execution_duration_seconds.labels(
            agent_type=agent_type
        ).observe(duration)
        
        # Track agent performance
        if agent_type not in self.agent_executions:
            self.agent_executions[agent_type] = {
                "total": 0, "successful": 0, "failed": 0, "avg_duration": 0.0
            }
            
        stats = self.agent_executions[agent_type]
        stats["total"] += 1
        if status == "success":
            stats["successful"] += 1
        else:
            stats["failed"] += 1
            
        # Update average duration
        stats["avg_duration"] = (
            (stats["avg_duration"] * (stats["total"] - 1) + duration) / stats["total"]
        )
        
    def record_database_operation(self, operation_type: str, collection: str, 
                                status: str, duration: float):
        """Record database operation metrics"""
        database_operations_total.labels(
            operation_type=operation_type,
            collection=collection,
            status=status
        ).inc()
        
        database_operation_duration_seconds.labels(
            operation_type=operation_type,
            collection=collection
        ).observe(duration)
        
    def record_cache_operation(self, operation_type: str, result: str):
        """Record cache operation metrics"""
        cache_operations_total.labels(
            operation_type=operation_type,
            result=result
        ).inc()
        
    def record_vector_search(self, company_id: str, search_type: str, 
                           status: str, duration: float):
        """Record vector search metrics"""
        vector_searches_total.labels(
            company_id=company_id,
            search_type=search_type,
            status=status
        ).inc()
        
        vector_search_duration_seconds.labels(
            search_type=search_type
        ).observe(duration)
        
    def update_system_metrics(self, active_connections: int, memory_usage: int):
        """Update system health metrics"""
        active_connections_gauge.set(active_connections)
        memory_usage_bytes.set(memory_usage)
        
    def update_crisis_cases(self, company_id: str, count: int):
        """Update crisis cases gauge"""
        crisis_cases_gauge.labels(company_id=company_id).set(count)
        
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        uptime = time.time() - self.start_time
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.request_count,
            "agent_executions": self.agent_executions,
            "performance_stats": self.performance_stats,
            "timestamp": datetime.utcnow().isoformat()
        }

# Global metrics collector instance
metrics_collector = MetricsCollector()

class StructuredLogger:
    """Structured logging with context management and correlation tracking"""
    
    def __init__(self):
        self.logger = structlog.get_logger()
        self.setup_logging()
        self.active_traces = {}  # Track active operation traces
        
    def setup_logging(self):
        """Configure structured logging"""
        # Configure standard logging
        logging_config = get_logging_config()
        logging.config.dictConfig(logging_config)
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.dev.ConsoleRenderer() if settings.LOG_FORMAT == "text" else structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, settings.LOG_LEVEL.upper())
            ),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
    def log_request(self, method: str, path: str, status_code: int, 
                   duration: float, user_id: Optional[str] = None):
        """Log HTTP request with structured data"""
        self.logger.info(
            "HTTP request completed",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration * 1000,
            user_id=user_id
        )
        
    def log_agent_execution(self, agent_type: str, crisis_case_id: str, 
                          company_id: str, status: str, duration: float,
                          error: Optional[str] = None):
        """Log agent execution with context"""
        log_data = {
            "agent_type": agent_type,
            "crisis_case_id": crisis_case_id,
            "company_id": company_id,
            "status": status,
            "duration_ms": duration * 1000
        }
        
        if error:
            log_data["error"] = error
            
        if status == "success":
            self.logger.info("Agent execution completed", **log_data)
        else:
            self.logger.error("Agent execution failed", **log_data)
            
    def log_database_operation(self, operation: str, collection: str, 
                             duration: float, status: str,
                             error: Optional[str] = None):
        """Log database operation"""
        log_data = {
            "database_operation": operation,
            "collection": collection,
            "duration_ms": duration * 1000,
            "status": status
        }
        
        if error:
            log_data["error"] = error
            
        if status == "success":
            self.logger.debug("Database operation completed", **log_data)
        else:
            self.logger.error("Database operation failed", **log_data)
            
    def log_security_event(self, event_type: str, user_id: Optional[str], 
                         company_id: Optional[str], details: Dict[str, Any]):
        """Log security-related events"""
        self.logger.warning(
            "Security event",
            event_type=event_type,
            user_id=user_id,
            company_id=company_id,
            **details
        )
    
    def start_trace(self, trace_id: str, operation_type: str, **context) -> None:
        """Start a new operation trace with correlation ID"""
        self.active_traces[trace_id] = {
            "operation_type": operation_type,
            "start_time": datetime.utcnow(),
            "context": context,
            "steps": []
        }
        
        self.logger.info(
            "Operation trace started",
            trace_id=trace_id,
            operation_type=operation_type,
            **context
        )
    
    def add_trace_step(self, trace_id: str, step_name: str, step_status: str, 
                       step_duration: Optional[float] = None, **step_data) -> None:
        """Add a step to an existing trace"""
        if trace_id in self.active_traces:
            step_info = {
                "step_name": step_name,
                "step_status": step_status,
                "timestamp": datetime.utcnow(),
                **step_data
            }
            
            if step_duration is not None:
                step_info["duration_ms"] = step_duration * 1000
            
            self.active_traces[trace_id]["steps"].append(step_info)
            
            self.logger.debug(
                "Trace step completed",
                trace_id=trace_id,
                step_name=step_name,
                step_status=step_status,
                **step_data
            )
    
    def end_trace(self, trace_id: str, final_status: str, **final_data) -> Dict[str, Any]:
        """End an operation trace and return trace summary"""
        if trace_id not in self.active_traces:
            self.logger.warning("Attempted to end non-existent trace", trace_id=trace_id)
            return {}
        
        trace_data = self.active_traces.pop(trace_id)
        end_time = datetime.utcnow()
        total_duration = (end_time - trace_data["start_time"]).total_seconds()
        
        trace_summary = {
            "trace_id": trace_id,
            "operation_type": trace_data["operation_type"],
            "total_duration_seconds": total_duration,
            "start_time": trace_data["start_time"].isoformat(),
            "end_time": end_time.isoformat(),
            "final_status": final_status,
            "total_steps": len(trace_data["steps"]),
            "successful_steps": len([s for s in trace_data["steps"] if s.get("step_status") == "success"]),
            "failed_steps": len([s for s in trace_data["steps"] if s.get("step_status") == "failed"]),
            "context": trace_data["context"],
            "steps": trace_data["steps"],
            **final_data
        }
        
        if final_status == "success":
            self.logger.info("Operation trace completed successfully", **trace_summary)
        else:
            self.logger.error("Operation trace completed with errors", **trace_summary)
        
        return trace_summary
    
    def log_agent_step(self, trace_id: str, agent_name: str, step_type: str, 
                      step_status: str, step_duration: float, 
                      crisis_case_id: str, company_id: str, 
                      input_summary: Optional[str] = None, 
                      output_summary: Optional[str] = None,
                      error_details: Optional[str] = None) -> None:
        """Log detailed agent execution step with full context"""
        log_data = {
            "trace_id": trace_id,
            "agent_name": agent_name,
            "step_type": step_type,
            "step_status": step_status,
            "duration_ms": step_duration * 1000,
            "crisis_case_id": crisis_case_id,
            "company_id": company_id
        }
        
        if input_summary:
            log_data["input_summary"] = input_summary
        if output_summary:
            log_data["output_summary"] = output_summary
        if error_details:
            log_data["error_details"] = error_details
        
        # Add to trace if it exists
        self.add_trace_step(
            trace_id, f"{agent_name}_{step_type}", step_status, 
            step_duration, **log_data
        )
        
        # Log step details
        if step_status == "success":
            self.logger.info("Agent step completed", **log_data)
        else:
            self.logger.error("Agent step failed", **log_data)
    
    def log_sub_agent_execution(self, trace_id: str, main_agent: str, sub_agent_name: str, 
                               sub_agent_index: int, execution_status: str, 
                               execution_duration: float, crisis_case_id: str, 
                               company_id: str, result_summary: Optional[str] = None,
                               error_details: Optional[str] = None) -> None:
        """Log sub-agent execution within a main agent workflow"""
        log_data = {
            "trace_id": trace_id,
            "main_agent": main_agent,
            "sub_agent_name": sub_agent_name,
            "sub_agent_index": sub_agent_index,
            "execution_status": execution_status,
            "duration_ms": execution_duration * 1000,
            "crisis_case_id": crisis_case_id,
            "company_id": company_id
        }
        
        if result_summary:
            log_data["result_summary"] = result_summary
        if error_details:
            log_data["error_details"] = error_details
        
        # Add to trace
        self.add_trace_step(
            trace_id, f"{main_agent}_sub_{sub_agent_index}_{sub_agent_name}", 
            execution_status, execution_duration, **log_data
        )
        
        if execution_status == "success":
            self.logger.info("Sub-agent execution completed", **log_data)
        else:
            self.logger.error("Sub-agent execution failed", **log_data)
    
    def log_workflow_phase(self, trace_id: str, phase_name: str, phase_status: str,
                          phase_duration: float, crisis_case_id: str, 
                          company_id: str, phase_output_id: Optional[str] = None,
                          error_details: Optional[str] = None) -> None:
        """Log major workflow phase completion"""
        log_data = {
            "trace_id": trace_id,
            "workflow_phase": phase_name,
            "phase_status": phase_status,
            "duration_ms": phase_duration * 1000,
            "crisis_case_id": crisis_case_id,
            "company_id": company_id
        }
        
        if phase_output_id:
            log_data["phase_output_id"] = phase_output_id
        if error_details:
            log_data["error_details"] = error_details
        
        # Add to trace
        self.add_trace_step(
            trace_id, f"workflow_phase_{phase_name}", phase_status, 
            phase_duration, **log_data
        )
        
        if phase_status == "success":
            self.logger.info("Workflow phase completed", **log_data)
        else:
            self.logger.error("Workflow phase failed", **log_data)
    
    def get_active_traces(self) -> Dict[str, Any]:
        """Get summary of all active traces"""
        current_time = datetime.utcnow()
        trace_summaries = {}
        
        for trace_id, trace_data in self.active_traces.items():
            elapsed_seconds = (current_time - trace_data["start_time"]).total_seconds()
            trace_summaries[trace_id] = {
                "operation_type": trace_data["operation_type"],
                "elapsed_seconds": elapsed_seconds,
                "steps_completed": len(trace_data["steps"]),
                "context": trace_data["context"],
                "is_long_running": elapsed_seconds > 300  # 5 minutes
            }
        
        return trace_summaries

# Global structured logger instance
structured_logger = StructuredLogger()

class PerformanceMonitor:
    """Performance monitoring and profiling"""
    
    def __init__(self):
        self.active_operations = {}
        self.performance_thresholds = {
            "http_request": 5.0,  # 5 seconds
            "agent_execution": 180.0,  # 3 minutes
            "database_query": 1.0,  # 1 second
            "vector_search": 2.0   # 2 seconds
        }
        
    async def monitor_operation(self, operation_type: str, operation_id: str,
                              threshold_override: Optional[float] = None):
        """Context manager for monitoring operation performance"""
        start_time = time.time()
        threshold = threshold_override or self.performance_thresholds.get(operation_type, 10.0)
        
        self.active_operations[operation_id] = {
            "type": operation_type,
            "start_time": start_time,
            "threshold": threshold
        }
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            
            # Check for performance issues
            if duration > threshold:
                structured_logger.logger.warning(
                    "Slow operation detected",
                    operation_type=operation_type,
                    operation_id=operation_id,
                    duration_seconds=duration,
                    threshold_seconds=threshold
                )
                
            # Clean up
            self.active_operations.pop(operation_id, None)
            
    def get_active_operations(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active operations"""
        current_time = time.time()
        
        for op_id, op_data in self.active_operations.items():
            op_data["elapsed_seconds"] = current_time - op_data["start_time"]
            op_data["over_threshold"] = op_data["elapsed_seconds"] > op_data["threshold"]
            
        return self.active_operations.copy()
        
    def check_for_stuck_operations(self) -> List[Dict[str, Any]]:
        """Identify operations that may be stuck"""
        current_time = time.time()
        stuck_operations = []
        
        for op_id, op_data in self.active_operations.items():
            elapsed = current_time - op_data["start_time"]
            if elapsed > op_data["threshold"] * 2:  # 2x threshold = stuck
                stuck_operations.append({
                    "operation_id": op_id,
                    "operation_type": op_data["type"],
                    "elapsed_seconds": elapsed,
                    "threshold_seconds": op_data["threshold"]
                })
                
        return stuck_operations

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

class HealthChecker:
    """System health monitoring"""
    
    def __init__(self):
        self.health_checks = {}
        self.last_check_time = None
        self.check_interval = 30  # 30 seconds
        
    async def register_health_check(self, name: str, check_func, timeout: float = 5.0):
        """Register a health check function"""
        self.health_checks[name] = {
            "check_func": check_func,
            "timeout": timeout,
            "last_result": None,
            "last_check": None
        }
        
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks"""
        results = {}
        overall_healthy = True
        
        for name, check_data in self.health_checks.items():
            try:
                # Run health check with timeout
                start_time = time.time()
                result = await asyncio.wait_for(
                    check_data["check_func"](),
                    timeout=check_data["timeout"]
                )
                
                duration = time.time() - start_time
                
                results[name] = {
                    "status": "healthy" if result else "unhealthy",
                    "duration_seconds": duration,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                if not result:
                    overall_healthy = False
                    
            except asyncio.TimeoutError:
                results[name] = {
                    "status": "timeout",
                    "duration_seconds": check_data["timeout"],
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": "Health check timed out"
                }
                overall_healthy = False
                
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": str(e)
                }
                overall_healthy = False
                
        self.last_check_time = datetime.utcnow()
        
        return {
            "overall_status": "healthy" if overall_healthy else "unhealthy",
            "checks": results,
            "timestamp": self.last_check_time.isoformat()
        }
        
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health check summary"""
        return {
            "total_checks": len(self.health_checks),
            "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
            "check_interval_seconds": self.check_interval
        }

# Global health checker instance
health_checker = HealthChecker()

async def setup_monitoring():
    """Initialize monitoring infrastructure"""
    logger.info("Setting up monitoring infrastructure")
    
    # Initialize OpenTelemetry if available
    if OTEL_AVAILABLE and settings.ENABLE_TRACING:
        trace.set_tracer_provider(TracerProvider())
        if settings.GOOGLE_CLOUD_PROJECT:
            # Set up GCP monitoring
            metrics.set_meter_provider(MeterProvider())
            
    # Register basic health checks
    await health_checker.register_health_check(
        "database", 
        lambda: True,  # Placeholder - would check DB connectivity
        timeout=5.0
    )
    
    await health_checker.register_health_check(
        "vector_db",
        lambda: True,  # Placeholder - would check Milvus connectivity
        timeout=5.0
    )
    
    logger.info("Monitoring infrastructure initialized")

def get_prometheus_metrics() -> str:
    """Get Prometheus metrics in text format"""
    return generate_latest(metrics_registry).decode('utf-8')

def get_system_stats() -> Dict[str, Any]:
    """Get comprehensive system statistics"""
    return {
        "metrics": metrics_collector.get_metrics_summary(),
        "active_operations": performance_monitor.get_active_operations(),
        "health": health_checker.get_health_summary(),
        "timestamp": datetime.utcnow().isoformat()
    }

# Export monitoring components
__all__ = [
    'metrics_collector',
    'structured_logger', 
    'performance_monitor',
    'health_checker',
    'setup_monitoring',
    'get_prometheus_metrics',
    'get_system_stats'
]