# MCP Server PRP — Brand PR Manager (Python Implementation)

## Purpose

This template is optimized for AI agents to implement a production-ready Model Context Protocol (MCP) server in Python that wraps open-source social listening tools, provides secure authenticated access, orchestrates LLM pipelines (Anthropic Claude), and writes alert/report outputs to persistent storage.

## Core Principles

1. **Context is King** — Provide complete brand and conversation context to the LLM and tools, including provenance and confidence.
2. **Validation Loops** — Automated tests using pytest, mypy type checking, and MCP Inspector validation.
3. **Security First** — OAuth-based authentication, RBAC, SQL/command injection protection, and least-privilege tool registration.
4. **Production Ready** — Logging, monitoring, graceful degradation, and deployable to both local Docker and cloud platforms.

## Technical Stack (Python-specific)

- **Web Framework**: FastAPI with Pydantic for data validation
- **Database**: SQLAlchemy with PostgreSQL
- **Queue System**: Redis with RQ for background jobs
- **Social Media Libraries**: 
  - Tweepy for Twitter
  - Instaloader for Instagram
  - PRAW for Reddit
  - feedparser for RSS
- **NLP & Content Processing**:
  - newspaper3k for article extraction
  - spaCy for local NLP processing
  - Anthropic Claude API for advanced analysis
- **Testing**: pytest with pytest-asyncio
- **Type Checking**: mypy with strict mode
- **Deployment**: Docker with docker-compose

## Project Structure

```
brand-pr-manager/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application entry point
│   ├── config.py               # Configuration management
│   ├── models/                 # SQLAlchemy models
│   │   ├── __init__.py
│   │   ├── mention.py
│   │   └── brand.py
│   ├── schemas/               # Pydantic models
│   │   ├── __init__.py
│   │   ├── mention.py
│   │   └── brand.py
│   ├── tools/                # MCP tool implementations
│   │   ├── __init__.py
│   │   ├── social.py         # Social media tools
│   │   └── reporter.py       # Reporting tools
│   ├── services/             # Business logic
│   │   ├── __init__.py
│   │   ├── anthropic.py      # Anthropic API client
│   │   └── enrichment.py     # Content enrichment
│   ├── workers/              # Background workers
│   │   ├── __init__.py
│   │   ├── classifier.py
│   │   └── monitor.py
│   └── utils/               # Utility functions
│       ├── __init__.py
│       ├── security.py
│       └── validation.py
├── tests/
│   ├── conftest.py          # pytest fixtures
│   ├── test_tools/         
│   └── test_services/
├── alembic/                 # Database migrations
├── docker/                  # Docker configurations
├── scripts/                 # Utility scripts
└── docs/                   # Documentation
```

## Core MCP Tools Implementation (Python)

```python
# Example tool registration pattern
@mcp_tool(
    name="search_tweets",
    description="Search tweets by query",
    schema=SearchTweetsSchema
)
async def search_tweets(query: str, since: Optional[str] = None, max_results: Optional[int] = None) -> MCPResponse:
    tweets = await twitter_service.search(query, since, max_results)
    return MCPResponse(content=[{"type": "json", "json": tweets}])
```

## Data Models (Pydantic)

```python
class SocialMention(BaseModel):
    id: str
    platform: Literal["twitter", "instagram", "reddit"]
    content: str
    author: str
    timestamp: datetime
    url: str
    metadata: Dict[str, Any]
    
class BrandProfile(BaseModel):
    id: str
    name: str
    keywords: List[str]
    watched_channels: List[str]
    alert_thresholds: Dict[str, float]
```

## Implementation Steps

1. **Project Setup**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install fastapi uvicorn sqlalchemy alembic redis pydantic[email] tweepy instaloader praw feedparser newspaper3k spacy anthropic
   ```

2. **Database Setup**
   ```bash
   # Initialize Alembic
   alembic init alembic
   
   # Create first migration
   alembic revision --autogenerate -m "initial"
   
   # Run migrations
   alembic upgrade head
   ```

3. **Docker Setup**
   ```yaml
   # docker-compose.yml
   version: '3'
   services:
     app:
       build: .
       environment:
         - DATABASE_URL=postgresql://postgres:postgres@db:5432/brand_pr
         - REDIS_URL=redis://redis:6379
       volumes:
         - ./data/reports:/data/reports
     db:
       image: postgres:15
     redis:
       image: redis:7
   ```

## Success Criteria

1. **Type Safety**
   - [ ] mypy --strict passes with no errors
   - [ ] All Pydantic models properly validate data

2. **Test Coverage**
   - [ ] >90% test coverage
   - [ ] Integration tests for each social platform
   - [ ] Mock tests for Anthropic API calls

3. **Performance**
   - [ ] API responses <500ms for non-LLM operations
   - [ ] Background jobs process within SLA
   - [ ] Rate limiting implemented for API endpoints

4. **Security**
   - [ ] All endpoints require authentication
   - [ ] SQL injection prevention verified
   - [ ] File path traversal prevention verified

## CI/CD Pipeline

```yaml
# GitHub Actions workflow
name: Python CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Type checking
        run: mypy app tests
      - name: Run tests
        run: pytest tests/ --cov=app
```

## Monitoring and Observability

1. **Logging**
   ```python
   # Using structlog for structured logging
   import structlog
   logger = structlog.get_logger()
   
   logger.info("mention_processed", 
               mention_id=mention.id,
               platform=mention.platform,
               processing_time=processing_time)
   ```

2. **Metrics**
   - Prometheus metrics via FastAPI endpoint
   - Key metrics:
     - Mentions processed per minute
     - Classification latency
     - Queue depth
     - API response times

3. **Alerts**
   - Configured in Grafana
   - Alert on:
     - High error rates
     - Processing delays
     - Queue backlog
     - API latency spikes

## Security Considerations

1. **Authentication**
   - GitHub OAuth2 implementation
   - JWT for API authentication
   - Rate limiting per API key

2. **Authorization**
   - Role-based access control (RBAC)
   - Tool-level permissions
   - Audit logging

3. **Data Security**
   - Encryption at rest
   - Secure file handling
   - Input validation

4. **Infrastructure**
   - Secure Docker configurations
   - Network isolation
   - Regular security updates

## Error Handling

```python
class BrandPRException(Exception):
    """Base exception for Brand PR Manager"""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

@app.exception_handler(BrandPRException)
async def brand_pr_exception_handler(request: Request, exc: BrandPRException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message}
    )
```

## Dependencies

```
# requirements.txt
fastapi>=0.100.0
uvicorn>=0.22.0
sqlalchemy>=2.0.0
alembic>=1.11.0
redis>=4.6.0
pydantic>=2.0.0
tweepy>=4.14.0
instaloader>=4.10.0
praw>=7.7.0
feedparser>=6.0.10
newspaper3k>=0.2.8
spacy>=3.6.0
anthropic>=0.3.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
mypy>=1.4.0
structlog>=23.1.0
```

---

This PRP provides a comprehensive blueprint for implementing the Brand PR Manager using Python and modern Python frameworks. The implementation focuses on type safety, testability, and production-readiness while maintaining clean architecture principles.
