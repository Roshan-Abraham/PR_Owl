"""
Mock ADK classes for development
Provides production-compatible mock implementations of Google ADK components
"""

from typing import Dict, Any, List, Optional, Protocol, runtime_checkable
from abc import ABC, abstractmethod
import asyncio
from datetime import datetime
import structlog

logger = structlog.get_logger()

# Check if real ADK is available
try:
    from google.adk import BaseAgent, SequentialAgent, ParallelAgent, WorkflowAgent, LlmAgent, AgentTool, Session, InvocationContext
    from google.adk.memory import VertexAiMemoryBankService
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    
    @runtime_checkable
    class BaseAgent(Protocol):
        """Base agent protocol matching ADK API"""
        
        @abstractmethod
        async def execute(self, context: 'InvocationContext') -> Dict[str, Any]:
            """Execute the agent with the given context"""
            pass
    
    class MockBaseAgent(ABC):
        """Mock base agent implementation"""
        def __init__(self, name: str = "", description: str = ""):
            self.name = name
            self.description = description
            self.metadata = {}
        
        @abstractmethod
        async def execute(self, context: 'InvocationContext') -> Dict[str, Any]:
            """Execute the agent with the given context"""
            pass
    
    class SequentialAgent(MockBaseAgent):
        """Mock sequential agent for development"""
        def __init__(self, agents: List[BaseAgent], name: str = "SequentialAgent"):
            super().__init__(name=name)
            self.agents = agents
            
        async def execute(self, context: 'InvocationContext') -> Dict[str, Any]:
            """Execute agents sequentially"""
            results = {}
            for i, agent in enumerate(self.agents):
                try:
                    logger.info(f"Executing sequential agent {i}: {getattr(agent, 'name', 'unnamed')}")
                    result = await agent.execute(context)
                    results[f"agent_{i}"] = result
                except Exception as e:
                    logger.error(f"Sequential agent {i} failed: {e}")
                    results[f"agent_{i}"] = {"error": str(e)}
            return {"sequential_results": results}
    
    class ParallelAgent(MockBaseAgent):
        """Mock parallel agent for development"""
        def __init__(self, agents: List[BaseAgent], name: str = "ParallelAgent"):
            super().__init__(name=name)
            self.agents = agents
            
        async def execute(self, context: 'InvocationContext') -> Dict[str, Any]:
            """Execute agents in parallel"""
            tasks = []
            for i, agent in enumerate(self.agents):
                # Create branched context for each agent
                branch_context = context.branch(f"parallel_branch_{i}")
                task = asyncio.create_task(
                    self._safe_execute(agent, branch_context, i),
                    name=f"parallel_agent_{i}"
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            processed_results = {}
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results[f"agent_{i}"] = {"error": str(result)}
                else:
                    processed_results[f"agent_{i}"] = result
            
            return {"parallel_results": processed_results}
        
        async def _safe_execute(self, agent: BaseAgent, context: 'InvocationContext', index: int) -> Dict[str, Any]:
            """Safely execute an agent with error handling"""
            try:
                logger.info(f"Executing parallel agent {index}: {getattr(agent, 'name', 'unnamed')}")
                return await agent.execute(context)
            except Exception as e:
                logger.error(f"Parallel agent {index} failed: {e}")
                raise
    
    class WorkflowAgent(MockBaseAgent):
        """Mock workflow agent for development"""
        def __init__(self, workflow_steps: List[Dict[str, Any]], name: str = "WorkflowAgent"):
            super().__init__(name=name)
            self.workflow_steps = workflow_steps
            
        async def execute(self, context: 'InvocationContext') -> Dict[str, Any]:
            """Execute workflow steps"""
            step_results = {}
            
            for i, step in enumerate(self.workflow_steps):
                try:
                    step_name = step.get("name", f"step_{i}")
                    logger.info(f"Executing workflow step {i}: {step_name}")
                    
                    # Mock step execution based on step type
                    if step.get("type") == "agent":
                        agent = step.get("agent")
                        if agent:
                            result = await agent.execute(context)
                        else:
                            result = {"mock": f"executed step {step_name}"}
                    else:
                        result = {"mock": f"executed step {step_name}"}
                    
                    step_results[step_name] = result
                    
                    # Update context with step result
                    context.session.state[f"step_{i}_result"] = result
                    
                except Exception as e:
                    logger.error(f"Workflow step {i} failed: {e}")
                    step_results[f"step_{i}"] = {"error": str(e)}
            
            return {"workflow_results": step_results}
    
    class LlmAgent(MockBaseAgent):
        """Mock LLM agent for development"""
        def __init__(self, name: str, prompt: str, tools: List['AgentTool'] = None):
            super().__init__(name=name)
            self.prompt = prompt
            self.tools = tools or []
            
        async def execute(self, context: 'InvocationContext') -> Dict[str, Any]:
            """Mock LLM execution"""
            logger.info(f"Executing LLM agent: {self.name}")
            
            # Simulate tool usage
            tool_results = {}
            for tool in self.tools:
                try:
                    tool_result = await tool.execute(context=context)
                    tool_results[tool.name] = tool_result
                except Exception as e:
                    tool_results[tool.name] = {"error": str(e)}
            
            return {
                "agent_name": self.name,
                "prompt_length": len(self.prompt),
                "tools_used": len(self.tools),
                "tool_results": tool_results,
                "mock_response": f"Mock LLM response for {self.name}",
                "session_context_keys": list(context.session.state.keys())
            }
    
    class AgentTool(ABC):
        """Mock agent tool base class"""
        def __init__(self, name: str = "", description: str = ""):
            self.name = name
            self.description = description
            
        @abstractmethod
        async def execute(self, **kwargs) -> Dict[str, Any]:
            """Execute the tool"""
            pass
    
    class Session:
        """Mock session implementation matching ADK API"""
        def __init__(self, session_id: str, user_id: Optional[str] = None):
            self.session_id = session_id
            self.user_id = user_id
            self.state: Dict[str, Any] = {}
            self.metadata: Dict[str, Any] = {}
            self.created_at = datetime.utcnow()
            self._memory_service: Optional['VertexAiMemoryBankService'] = None
            
        def get(self, key: str, default: Any = None) -> Any:
            """Get value from session state"""
            return self.state.get(key, default)
            
        def set(self, key: str, value: Any) -> None:
            """Set value in session state"""
            self.state[key] = value
            
        def update(self, data: Dict[str, Any]) -> None:
            """Update session state with dictionary"""
            self.state.update(data)
            
        def clear(self) -> None:
            """Clear session state"""
            self.state.clear()
            
        def set_memory_service(self, memory_service: 'VertexAiMemoryBankService') -> None:
            """Set the memory service for this session"""
            self._memory_service = memory_service
            
        def get_memory_service(self) -> Optional['VertexAiMemoryBankService']:
            """Get the memory service for this session"""
            return self._memory_service
    
    class InvocationContext:
        """Mock invocation context matching ADK API"""
        def __init__(self, session: Session):
            self.session = session
            self.metadata: Dict[str, Any] = {}
            self.branch_id: Optional[str] = None
            
        def branch(self, branch_id: str) -> 'InvocationContext':
            """Create a branched context for parallel execution"""
            new_context = InvocationContext(self.session)
            new_context.metadata = self.metadata.copy()
            new_context.metadata["parent_branch"] = self.branch_id
            new_context.branch_id = branch_id
            return new_context
            
        def get_branch_id(self) -> Optional[str]:
            """Get the current branch ID"""
            return self.branch_id
            
        def is_branched(self) -> bool:
            """Check if this context is branched"""
            return self.branch_id is not None
    
    class VertexAiMemoryBankService:
        """Mock memory bank service matching ADK API"""
        def __init__(self, project_id: str, location: str = "us-central1"):
            self.project_id = project_id
            self.location = location
            self.memories: Dict[str, List[Dict[str, Any]]] = {}
            
        async def add_session_to_memory(self, session: Session, 
                                       filters: Optional[Dict[str, str]] = None) -> None:
            """Mock memory addition"""
            session_key = session.session_id
            if session_key not in self.memories:
                self.memories[session_key] = []
            
            memory_entry = {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "state_snapshot": session.state.copy(),
                "metadata": session.metadata.copy(),
                "filters": filters or {},
                "timestamp": datetime.utcnow().isoformat(),
                "memory_id": f"mem_{len(self.memories[session_key])}"
            }
            self.memories[session_key].append(memory_entry)
            logger.info(f"Added memory entry for session {session_key}")
            
        async def search_memory(self, query: str, filters: Optional[Dict[str, str]] = None,
                               top_k: int = 10, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
            """Mock memory search with semantic-like behavior"""
            results = []
            
            # If session_id provided, search only that session
            if session_id and session_id in self.memories:
                memories_to_search = {session_id: self.memories[session_id]}
            else:
                memories_to_search = self.memories
            
            # Simple mock search - in real implementation would use semantic search
            for sid, memories in memories_to_search.items():
                for memory in memories:
                    # Apply filters if provided
                    if filters:
                        match = all(
                            memory.get("filters", {}).get(k) == v 
                            for k, v in filters.items()
                        )
                        if not match:
                            continue
                    
                    # Simple text matching for mock implementation
                    memory_text = str(memory.get("state_snapshot", {}))
                    if query.lower() in memory_text.lower():
                        results.append({
                            **memory,
                            "relevance_score": 0.8  # Mock relevance score
                        })
            
            # Sort by timestamp (most recent first) and return top_k
            results.sort(key=lambda x: x["timestamp"], reverse=True)
            return results[:top_k]
            
        async def delete_memory(self, session_id: str, memory_id: Optional[str] = None) -> None:
            """Mock memory deletion"""
            if session_id in self.memories:
                if memory_id:
                    # Delete specific memory
                    self.memories[session_id] = [
                        m for m in self.memories[session_id] 
                        if m.get("memory_id") != memory_id
                    ]
                else:
                    # Delete all memories for session
                    del self.memories[session_id]
                logger.info(f"Deleted memory for session {session_id}")


# Export all classes for easy import
__all__ = [
    'ADK_AVAILABLE',
    'BaseAgent', 
    'SequentialAgent', 
    'ParallelAgent',
    'WorkflowAgent',
    'LlmAgent', 
    'AgentTool', 
    'Session', 
    'InvocationContext',
    'VertexAiMemoryBankService'
]