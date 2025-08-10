# Anthropic API Usage Guide

## Overview

This guide covers best practices for using the Anthropic Claude API in the context of the Brand PR Manager MCP server, focusing on classification, summarization, and tool invocation patterns.

## Setup

```python
from anthropic import Anthropic
from typing import List, Dict, Any

class AnthropicClient:
    def __init__(
        self,
        api_key: str,
        mcp_server_url: str = None
    ):
        self.client = Anthropic(
            api_key=api_key,
            default_headers={
                'anthropic-beta': 'mcp-client-2025-04-04'
            } if mcp_server_url else {}
        )
        if mcp_server_url:
            self.client.mcp_servers = [mcp_server_url]
```

## Classification Pattern

```python
async def classify_mention(
    self,
    mention: Dict[str, Any],
    brand_profile: Dict[str, Any],
    recent_context: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Classify a social media mention using Claude"""
    
    system_prompt = """
    You are a PR classification assistant. Output STRICT JSON only.
    Analyze the mention and determine:
    1. Issue type
    2. Severity (0-1)
    3. Recommended action
    4. Confidence score
    
    Return JSON format:
    {
        "issue_type": str,
        "severity": float,
        "explanation": str,
        "recommended_action": str,
        "channels": List[str],
        "timing": str,
        "confidence": float,
        "human_review": bool
    }
    """
    
    user_message = {
        "brand_profile": brand_profile,
        "mention": mention,
        "recent_context": recent_context or []
    }
    
    response = await self.client.messages.create(
        model="claude-3-sonnet-20240229",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_message)}
        ],
        temperature=0.1,
        max_tokens=1000
    )
    
    return json.loads(response.content[0].text)
```

## Summarization Pattern

```python
async def summarize_mentions(
    self,
    mentions: List[Dict[str, Any]],
    max_length: int = 500
) -> str:
    """Generate a concise summary of multiple mentions"""
    
    system_prompt = """
    Summarize the key points from these social media mentions.
    Focus on:
    1. Main themes/issues
    2. Sentiment trends
    3. Notable influencers
    4. Potential risks
    
    Keep the summary concise and actionable.
    """
    
    response = await self.client.messages.create(
        model="claude-3-sonnet-20240229",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(mentions)}
        ],
        temperature=0.3,
        max_tokens=max_length
    )
    
    return response.content[0].text
```

## Response Generation Pattern

```python
async def generate_response(
    self,
    mention: Dict[str, Any],
    brand_profile: Dict[str, Any],
    tone: str = "professional"
) -> str:
    """Generate an appropriate response to a mention"""
    
    system_prompt = f"""
    You are a PR response generator for {brand_profile['name']}.
    Generate a {tone} response that:
    1. Addresses the key concerns
    2. Maintains brand voice
    3. Is authentic and helpful
    4. Follows social media best practices
    
    Do not include hashtags or @mentions unless specifically requested.
    """
    
    response = await self.client.messages.create(
        model="claude-3-sonnet-20240229",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(mention)}
        ],
        temperature=0.7,
        max_tokens=280  # Twitter-length for social responses
    )
    
    return response.content[0].text
```

## Best Practices

### 1. Prompt Engineering

- Use system prompts to set clear roles and constraints
- Request specific output formats (e.g., JSON)
- Include examples for complex tasks
- Set appropriate temperature for the task

### 2. Error Handling

```python
class AnthropicError(Exception):
    """Base exception for Anthropic API errors"""
    pass

async def safe_classify(
    self,
    mention: Dict[str, Any],
    retries: int = 3
) -> Dict[str, Any]:
    """Safely classify with retries and error handling"""
    
    for attempt in range(retries):
        try:
            result = await self.classify_mention(mention)
            
            # Validate response format
            if not self._validate_classification(result):
                raise AnthropicError("Invalid classification format")
                
            return result
            
        except Exception as e:
            if attempt == retries - 1:
                raise AnthropicError(f"Classification failed: {str(e)}")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

### 3. Response Validation

```python
def _validate_classification(
    self,
    result: Dict[str, Any]
) -> bool:
    """Validate classification response format and values"""
    
    required_fields = {
        "issue_type": str,
        "severity": float,
        "confidence": float
    }
    
    try:
        # Check required fields and types
        for field, field_type in required_fields.items():
            if field not in result:
                return False
            if not isinstance(result[field], field_type):
                return False
                
        # Validate value ranges
        if not 0 <= result["severity"] <= 1:
            return False
        if not 0 <= result["confidence"] <= 1:
            return False
            
        return True
        
    except Exception:
        return False
```

### 4. Rate Limiting

```python
from asyncio import Semaphore
from functools import wraps

def rate_limit(max_concurrent: int = 5):
    """Rate limiting decorator for API calls"""
    sem = Semaphore(max_concurrent)
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with sem:
                return await func(*args, **kwargs)
        return wrapper
    return decorator

# Usage
@rate_limit(max_concurrent=5)
async def classify_mention(self, mention: Dict[str, Any]):
    # Implementation
    pass
```

## Performance Optimization

### 1. Batching

```python
async def batch_classify(
    self,
    mentions: List[Dict[str, Any]],
    batch_size: int = 5
) -> List[Dict[str, Any]]:
    """Classify multiple mentions in batches"""
    
    results = []
    for i in range(0, len(mentions), batch_size):
        batch = mentions[i:i + batch_size]
        tasks = [self.classify_mention(mention) for mention in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
    
    return results
```

### 2. Caching

```python
from functools import lru_cache
from datetime import datetime, timedelta

class CachedAnthropicClient(AnthropicClient):
    @lru_cache(maxsize=1000)
    def _get_cached_classification(
        self,
        mention_id: str,
        content_hash: str
    ) -> Dict[str, Any]:
        """Cache classification results"""
        pass
        
    async def classify_mention(
        self,
        mention: Dict[str, Any],
        use_cache: bool = True
    ) -> Dict[str, Any]:
        if use_cache:
            mention_id = mention.get("id")
            content_hash = self._hash_content(mention)
            cached = self._get_cached_classification(mention_id, content_hash)
            if cached:
                return cached
                
        result = await super().classify_mention(mention)
        if use_cache:
            self._get_cached_classification.cache_clear()
        return result
```

## Monitoring & Logging

```python
import structlog
from prometheus_client import Counter, Histogram

logger = structlog.get_logger()

# Metrics
CLAUDE_REQUESTS = Counter(
    'claude_requests_total',
    'Total Claude API requests',
    ['operation']
)
CLAUDE_LATENCY = Histogram(
    'claude_request_latency_seconds',
    'Claude API request latency',
    ['operation']
)

def log_claude_call(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        operation = func.__name__
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Update metrics
            CLAUDE_REQUESTS.labels(operation).inc()
            CLAUDE_LATENCY.labels(operation).observe(duration)
            
            # Structured logging
            logger.info(
                "claude_request_complete",
                operation=operation,
                duration=duration,
                status="success"
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "claude_request_failed",
                operation=operation,
                error=str(e),
                duration=time.time() - start_time
            )
            raise
            
    return wrapper
```

This documentation provides comprehensive patterns and best practices for integrating the Anthropic Claude API into the Brand PR Manager MCP server, focusing on robustness, performance, and monitoring.
