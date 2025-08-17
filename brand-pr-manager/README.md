# Crisis Management System

A production-ready FastAPI application implementing a comprehensive crisis simulation and response system using Firestore, Vertex AI agents, and vector search capabilities.

## 🏗️ Architecture Overview

The system implements a multi-agent architecture with three main orchestrators and 21 specialized sub-agents:

- **FastAPI** for REST API endpoints and async request handling
- **Google Cloud Firestore** as the primary database with connection pooling
- **ADK (Agent Development Kit)** for agent orchestration using Vertex AI
- **Milvus** as the vector database for case study retrieval
- **Vertex AI Memory Bank** for long-term semantic memory
- **FastMap-optimized MCP tools** for high-performance database operations

### Agent Architecture

#### 🔍 Context Collector (Agent A) - 7 Sub-agents

1. **CompanyProfileAgent** - Company data validation and completeness scoring
2. **StakeholderMappingAgent** - Relationship intelligence and influence analysis
3. **EventContextAgent** - Temporal context and timeline reconstruction
4. **HistoricalPatternAgent** - Crisis history analysis and pattern recognition
5. **ExternalSignalsAgent** - Market intelligence and social sentiment
6. **KnowledgeBaseAgent** - Internal knowledge retrieval with vector search
7. **SnapshotSynthesizerAgent** - Data integration and quality validation

#### ⚡ Classification Agent (Agent B) - 7 Sub-agents

1. **SeverityAssessmentAgent** - Crisis magnitude and benchmarking
2. **ImpactPredictionAgent** - Consequence analysis and scenario modeling
3. **StakeholderExposureAgent** - Stakeholder risk and communication priorities
4. **TimelineAnalysisAgent** - Temporal dynamics and response windows
5. **CompetitiveContextAgent** - Market positioning and brand impact
6. **LegalComplianceAgent** - Regulatory risk and compliance validation
7. **ScorecardSynthesizerAgent** - Multi-dimensional score integration

#### 💡 Recommendation Agent (Agent C) - 7 Sub-agents

1. **HistoricalCaseSearchAgent** - Precedent research and outcome analysis
2. **ScenarioModelingAgent** - Strategic options and response strategies
3. **StakeholderStrategyAgent** - Communication planning and messaging
4. **ResourceOptimizationAgent** - Implementation planning and budgeting
5. **RiskMitigationAgent** - Contingency planning and fallback options
6. **ComplianceValidatorAgent** - Legal and regulatory validation
7. **RecommendationSynthesizerAgent** - Strategy integration and execution roadmap

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Google Cloud Project with Firestore enabled
- Google Cloud Service Account credentials
- Docker (optional, for Milvus)

### Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd brand-pr-manager
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Configure Google Cloud credentials**

```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

5. **Start the application**

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Setup (Optional)

For local development with Milvus, first download the `docker-compose.yml` file:

```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.4.4/milvus-standalone-docker-compose.yml -O docker-compose.yml
```

Then, start Milvus and its dependencies:

```bash
docker compose up -d
```

**Note:** This uses `docker compose` (with a space), which is the current standard.

# Verify Milvus is running

curl http://localhost:9091/healthz

## 📋 API Endpoints

### Crisis Management

- `POST /companies/{company_id}/simulate` - Create and start crisis simulation for a company
- `GET /companies/{company_id}/crises/{crisis_case_id}` - Get crisis case details
- `GET /companies/{company_id}/crises/{crisis_case_id}/snapshot` - Get latest crisis snapshot (stored as an Artifact)
- `POST /companies/{company_id}/crises/{crisis_case_id}/classify` - Trigger classification manually
- `POST /companies/{company_id}/crises/{crisis_case_id}/recommend` - Trigger recommendations manually

### Company Management

- `GET /companies/{company_id}/dashboard` - Get company dashboard
- `GET /companies/{company_id}/crises` - List company crises
- `GET /companies/{company_id}/profile` - Get company profile
- `POST /companies/{company_id}/profile` - Update company profile

### System Health

- `GET /health` - Health check endpoint
- `GET /metrics` - Prometheus metrics

## 🗄️ Database Schema

### Collections Structure

```
companies/{company_id}
├── dashboard/summary          # Real-time metrics
├── details/profile           # Company information
├── events/{event_id}         # Company events
├── relations/{relation_id}   # Stakeholder relationships
├── templates/{template_id}   # Simulation templates
└── knowledge_base/{kb_id}    # Internal case studies

Company/{company_id}/Crises/{crisis_case_id}
├── Artifacts/{artifact_id}   # Multi-type artifacts (artifact_type: snapshot|scorecard|recommendation)
├── logs/{log_id}            # Agent execution logs
└── agent_sessions/{session_id} # ADK session state

agent_runs/{run_id}          # Global agent tracking
vector_metadata/{collection}/{object_id} # Vector sync data
dashboards/{company_id}      # Top-level dashboard data
```

### Key Data Models

- **CrisisCase** - Main crisis record with status tracking
- **CrisisSnapshot** - Comprehensive context data
- **Scorecard** - Multi-dimensional risk assessment
- **Recommendation** - Strategic response plan
- **CompanyProfile** - Company master data
- **AgentLog** - Execution audit trail

## 🛠️ Configuration

### Environment Variables

| Variable                  | Description                | Default            |
| ------------------------- | -------------------------- | ------------------ |
| `GOOGLE_CLOUD_PROJECT`    | GCP project ID             | Required           |
| `FIRESTORE_DATABASE`      | Firestore database name    | `(default)`        |
| `MILVUS_HOST`             | Milvus server host         | `localhost`        |
| `MILVUS_PORT`             | Milvus server port         | `19530`            |
| `EMBEDDING_MODEL`         | Sentence transformer model | `all-MiniLM-L6-v2` |
| `LOG_LEVEL`               | Logging level              | `INFO`             |
| `AGENT_EXECUTION_TIMEOUT` | Agent timeout (seconds)    | `180`              |
| `MAX_CONCURRENT_AGENTS`   | Max concurrent agents      | `10`               |

### Feature Flags

- `ENABLE_VECTOR_SEARCH` - Enable vector similarity search
- `ENABLE_EXTERNAL_SIGNALS` - Enable external market intelligence
- `ENABLE_DASHBOARD_UPDATES` - Enable real-time dashboard updates
- `ENABLE_AUDIT_LOGGING` - Enable comprehensive audit logging

## 📊 Monitoring & Observability

### Prometheus Metrics

- **HTTP Requests**: Request count, duration, status codes
- **Agent Executions**: Success/failure rates, execution times
- **Database Operations**: Query performance, connection pool status
- **Vector Searches**: Search latency, cache hit rates
- **System Health**: Memory usage, active connections

### Structured Logging

All operations are logged with structured data including:

- Request/response correlation IDs
- Agent execution context
- Performance metrics
- Error details and stack traces

### Health Checks

- Database connectivity
- Vector database status
- Agent service availability
- Cache system health

## 🔒 Security

### Multi-tenancy

- All operations are company-scoped
- Data isolation enforced at database level
- Session-based access control
- Audit trails for all actions

### Access Control

- Service account based authentication
- Role-based permissions for agents
- API key management for external services
- Firestore security rules enforcement

## 🧪 Testing

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-mock

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

### Test Categories

- **Unit Tests** - Individual component testing
- **Integration Tests** - Database and API testing
- **Agent Tests** - Agent workflow validation
- **Performance Tests** - Load and stress testing

## 📈 Performance Optimization

### Database Performance

- **Connection Pooling** - 5-20 concurrent connections
- **Intelligent Caching** - Adaptive TTL based on data volatility
- **Query Optimization** - Compound indexes and query planning
- **Batch Operations** - High-throughput write operations

### Vector Search Performance

- **Embedding Caching** - 1-hour TTL for reused embeddings
- **Result Diversification** - Balanced results across source types
- **Hybrid Search** - Combined vector and keyword search
- **Connection Pooling** - Milvus connection management

### Agent Optimization

- **Concurrent Execution** - Parallel sub-agent processing
- **Caching Strategy** - Context and result caching
- **Performance Monitoring** - Execution time tracking
- **Resource Limits** - Memory and timeout controls

## 🚧 Development Status

### ✅ Completed Features (65%)

- FastAPI application with all endpoints
- Firestore connection pooling and caching
- Context Collector Agent with 7 sub-agents
- Vector search integration with Milvus
- Comprehensive monitoring and logging
- Data models and schema validation
- MCP tools with performance optimization

### 🔄 In Progress

- Classification Agent implementation
- Recommendation Agent implementation
- ADK integration and session management
- Complete test suite development

### 📋 Planned Features

- Docker development environment
- CI/CD pipeline with automated testing
- Sample crisis simulation data
- Production deployment guides
- API documentation with examples

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Run pre-commit hooks: `pre-commit install`
5. Write tests for new features
6. Submit a pull request

### Code Standards

- **Black** for code formatting
- **isort** for import sorting
- **mypy** for type checking
- **pytest** for testing
- **Structured logging** for all operations

## 📚 Documentation

- [API Documentation](docs/api.md) - Complete API reference
- [Agent Development Guide](docs/agents.md) - Creating custom agents
- [Deployment Guide](docs/deployment.md) - Production deployment
- [Configuration Reference](docs/configuration.md) - All settings
- [Troubleshooting Guide](docs/troubleshooting.md) - Common issues

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Google Cloud Firestore](https://cloud.google.com/firestore/docs)
- [Vertex AI Agent Builder](https://cloud.google.com/agent-builder)
- [Milvus Vector Database](https://milvus.io/docs)
- [ADK Documentation](https://google.github.io/adk-docs/)

## 📞 Support

For questions, issues, or contributions:

- **Issues**: [GitHub Issues](../../issues)
- **Discussions**: [GitHub Discussions](../../discussions)
- **Documentation**: [Project Wiki](../../wiki)

---

**Built with ❤️ using FastAPI, Firestore, Vertex AI, and Milvus**
