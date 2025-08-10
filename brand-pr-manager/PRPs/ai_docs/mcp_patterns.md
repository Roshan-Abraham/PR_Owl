# MCP Development Patterns (Python)

## Core Patterns

### Tool Registration

```python
from mcp import MCPServer, MCPResponse
from pydantic import BaseModel

class ToolInput(BaseModel):
    param1: str
    param2: int

@mcp_tool(
    name="tool_name",
    description="Tool description",
    schema=ToolInput
)
async def my_tool(param1: str, param2: int) -> MCPResponse:
    # Tool implementation
    result = await process(param1, param2)
    return MCPResponse(content=[{"type": "json", "json": result}])
```

### Error Handling

```python
class MCPToolError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

def handle_tool_error(func):
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except MCPToolError as e:
            return MCPResponse(
                error={"message": e.message, "code": e.status_code}
            )
        except Exception as e:
            return MCPResponse(
                error={"message": str(e), "code": 500}
            )
    return wrapper
```

### Authentication & Authorization

```python
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_token(token: str = Depends(oauth2_scheme)):
    if not is_valid_token(token):
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials"
        )
    return get_user_from_token(token)

def require_permission(permission: str):
    def decorator(func):
        async def wrapper(user = Depends(verify_token), *args, **kwargs):
            if not has_permission(user, permission):
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission denied: {permission} required"
                )
            return await func(user=user, *args, **kwargs)
        return wrapper
    return decorator
```

### Database Safety

```python
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

async def safe_execute_query(session, query: str, params: dict = None):
    """Execute SQL query with safety checks"""
    try:
        # Validate query before execution
        if not is_safe_query(query):
            raise ValueError("Unsafe SQL query detected")
            
        result = await session.execute(text(query), params)
        return result
    except SQLAlchemyError as e:
        logger.error("Database error", error=str(e), query=query)
        raise MCPToolError("Database operation failed", 500)
```

## Best Practices

1. **Input Validation**
   - Always use Pydantic models for input validation
   - Implement custom validators for complex rules
   - Sanitize file paths and SQL queries

2. **Error Handling**
   - Use custom exception classes
   - Provide meaningful error messages
   - Include error codes for client handling

3. **Performance**
   - Use connection pooling
   - Implement caching where appropriate
   - Use background tasks for long operations

4. **Security**
   - Validate all inputs
   - Use parameterized queries
   - Implement rate limiting
   - Log security events

## Common Gotchas

1. **File Operations**
```python
import os
from pathlib import Path

def safe_join_paths(base: str, *paths: str) -> str:
    """Safely join paths preventing directory traversal"""
    base_path = Path(base).resolve()
    joined = base_path.joinpath(*paths).resolve()
    if not str(joined).startswith(str(base_path)):
        raise ValueError("Path traversal detected")
    return str(joined)
```

2. **Rate Limiting**
```python
from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        {"error": "Rate limit exceeded"},
        status_code=429
    )
```

3. **Long-Running Operations**
```python
from fastapi import BackgroundTasks

async def process_in_background(data: dict):
    # Long-running process
    pass

@app.post("/process")
async def process_endpoint(
    data: dict,
    background_tasks: BackgroundTasks
):
    # Queue the task
    background_tasks.add_task(process_in_background, data)
    return {"status": "processing"}
```

## Testing Patterns

```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_social_tool():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/mcp/tools/search_tweets",
            json={"query": "test"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
```

## Monitoring and Logging

```python
import structlog
from prometheus_client import Counter, Histogram

# Metrics
REQUESTS = Counter('mcp_requests_total', 'Total MCP requests')
LATENCY = Histogram('mcp_request_latency_seconds', 'Request latency')

# Structured logging
logger = structlog.get_logger()

def log_request(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(
                "request_processed",
                duration=duration,
                status="success"
            )
            return result
        except Exception as e:
            logger.error(
                "request_failed",
                error=str(e),
                duration=time.time() - start_time
            )
            raise
    return wrapper
```
