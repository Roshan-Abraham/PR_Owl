"""
Configuration management for Crisis Management System
Handles environment variables, settings, and feature flags
"""

import os
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Application settings
    APP_NAME: str = Field(default="Crisis Management System", env="APP_NAME")
    APP_VERSION: str = Field(default="1.0.0", env="APP_VERSION")
    DEBUG: bool = Field(default=False, env="DEBUG")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")

    # Server settings
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=1, env="WORKERS")

    # Google Cloud / Firebase settings
    GOOGLE_CLOUD_PROJECT: str = Field(default="", env="GOOGLE_CLOUD_PROJECT")
    GOOGLE_APPLICATION_CREDENTIALS: str = Field(
        default="", env="GOOGLE_APPLICATION_CREDENTIALS")
    FIRESTORE_DATABASE: str = Field(
        default="(default)", env="FIRESTORE_DATABASE")

    # Firestore connection pool settings
    FIRESTORE_MIN_CONNECTIONS: int = Field(
        default=5, env="FIRESTORE_MIN_CONNECTIONS")
    FIRESTORE_MAX_CONNECTIONS: int = Field(
        default=20, env="FIRESTORE_MAX_CONNECTIONS")

    # Vector database settings (Milvus)
    MILVUS_HOST: str = Field(default="localhost", env="MILVUS_HOST")
    MILVUS_PORT: str = Field(default="19530", env="MILVUS_PORT")
    MILVUS_USER: str = Field(default="", env="MILVUS_USER")
    MILVUS_PASSWORD: str = Field(default="", env="MILVUS_PASSWORD")
    MILVUS_SECURE: bool = Field(default=False, env="MILVUS_SECURE")

    # Embedding settings
    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    EMBEDDING_DIMENSION: int = Field(default=384, env="EMBEDDING_DIMENSION")

    # Cache settings (Redis)
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # 1 hour

    # ADK / Vertex AI settings
    VERTEX_AI_LOCATION: str = Field(
        default="us-central1", env="VERTEX_AI_LOCATION")
    VERTEX_AI_MODEL: str = Field(default="gemini-pro", env="VERTEX_AI_MODEL")
    ADK_AGENT_BUILDER_LOCATION: str = Field(
        default="global", env="ADK_AGENT_BUILDER_LOCATION")

    # Security settings
    SECRET_KEY: str = Field(
        default="dev-secret-key-change-in-production", env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    ALLOWED_HOSTS: list = Field(default=["*"], env="ALLOWED_HOSTS")

    # CORS settings
    CORS_ORIGINS: list = Field(default=["*"], env="CORS_ORIGINS")
    CORS_ALLOW_CREDENTIALS: bool = Field(
        default=True, env="CORS_ALLOW_CREDENTIALS")

    # Logging settings
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")  # json or text

    # Monitoring settings
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=8001, env="METRICS_PORT")
    ENABLE_TRACING: bool = Field(default=False, env="ENABLE_TRACING")
    JAEGER_ENDPOINT: str = Field(default="", env="JAEGER_ENDPOINT")

    # Performance settings
    MAX_REQUEST_SIZE: int = Field(
        default=16777216, env="MAX_REQUEST_SIZE")  # 16MB
    REQUEST_TIMEOUT: int = Field(
        default=300, env="REQUEST_TIMEOUT")  # 5 minutes

    # Agent execution settings
    AGENT_EXECUTION_TIMEOUT: int = Field(
        default=180, env="AGENT_EXECUTION_TIMEOUT")  # 3 minutes
    MAX_CONCURRENT_AGENTS: int = Field(default=10, env="MAX_CONCURRENT_AGENTS")
    AGENT_RETRY_ATTEMPTS: int = Field(default=3, env="AGENT_RETRY_ATTEMPTS")

    # Crisis simulation settings
    DEFAULT_SIMULATION_TIMEOUT: int = Field(
        default=120, env="DEFAULT_SIMULATION_TIMEOUT")  # 2 minutes
    MAX_CRISIS_CASES_PER_COMPANY: int = Field(
        default=1000, env="MAX_CRISIS_CASES_PER_COMPANY")

    # Feature flags
    ENABLE_VECTOR_SEARCH: bool = Field(
        default=True, env="ENABLE_VECTOR_SEARCH")
    ENABLE_EXTERNAL_SIGNALS: bool = Field(
        default=True, env="ENABLE_EXTERNAL_SIGNALS")
    ENABLE_DASHBOARD_UPDATES: bool = Field(
        default=True, env="ENABLE_DASHBOARD_UPDATES")
    ENABLE_AUDIT_LOGGING: bool = Field(
        default=True, env="ENABLE_AUDIT_LOGGING")

    # External API settings (for ExternalSignalsAgent)
    NEWS_API_KEY: str = Field(default="", env="NEWS_API_KEY")
    TWITTER_API_KEY: str = Field(default="", env="TWITTER_API_KEY")
    TWITTER_API_SECRET: str = Field(default="", env="TWITTER_API_SECRET")

    # Database settings
    DATABASE_QUERY_TIMEOUT: int = Field(
        default=30, env="DATABASE_QUERY_TIMEOUT")  # seconds
    DATABASE_CONNECTION_TIMEOUT: int = Field(
        default=10, env="DATABASE_CONNECTION_TIMEOUT")  # seconds

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_database_config() -> Dict[str, Any]:
    """Get database configuration"""
    return {
        "project_id": settings.GOOGLE_CLOUD_PROJECT,
        "database": settings.FIRESTORE_DATABASE,
        "min_connections": settings.FIRESTORE_MIN_CONNECTIONS,
        "max_connections": settings.FIRESTORE_MAX_CONNECTIONS,
        "query_timeout": settings.DATABASE_QUERY_TIMEOUT,
        "connection_timeout": settings.DATABASE_CONNECTION_TIMEOUT,
    }


def get_vector_db_config() -> Dict[str, Any]:
    """Get vector database configuration"""
    return {
        "host": settings.MILVUS_HOST,
        "port": settings.MILVUS_PORT,
        "user": settings.MILVUS_USER,
        "password": settings.MILVUS_PASSWORD,
        "secure": settings.MILVUS_SECURE,
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_dimension": settings.EMBEDDING_DIMENSION,
    }


def get_agent_config() -> Dict[str, Any]:
    """Get agent execution configuration"""
    return {
        "execution_timeout": settings.AGENT_EXECUTION_TIMEOUT,
        "max_concurrent": settings.MAX_CONCURRENT_AGENTS,
        "retry_attempts": settings.AGENT_RETRY_ATTEMPTS,
        "vertex_ai_location": settings.VERTEX_AI_LOCATION,
        "vertex_ai_model": settings.VERTEX_AI_MODEL,
        "adk_location": settings.ADK_AGENT_BUILDER_LOCATION,
    }


def get_cache_config() -> Dict[str, Any]:
    """Get cache configuration"""
    return {
        "redis_url": settings.REDIS_URL,
        "default_ttl": settings.CACHE_TTL,
    }


def get_security_config() -> Dict[str, Any]:
    """Get security configuration"""
    return {
        "secret_key": settings.SECRET_KEY,
        "token_expire_minutes": settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        "allowed_hosts": settings.ALLOWED_HOSTS,
        "cors_origins": settings.CORS_ORIGINS,
        "cors_allow_credentials": settings.CORS_ALLOW_CREDENTIALS,
    }


def get_monitoring_config() -> Dict[str, Any]:
    """Get monitoring and observability configuration"""
    return {
        "enable_metrics": settings.ENABLE_METRICS,
        "metrics_port": settings.METRICS_PORT,
        "enable_tracing": settings.ENABLE_TRACING,
        "jaeger_endpoint": settings.JAEGER_ENDPOINT,
        "log_level": settings.LOG_LEVEL,
        "log_format": settings.LOG_FORMAT,
    }


def get_feature_flags() -> Dict[str, bool]:
    """Get all feature flags"""
    return {
        "vector_search": settings.ENABLE_VECTOR_SEARCH,
        "external_signals": settings.ENABLE_EXTERNAL_SIGNALS,
        "dashboard_updates": settings.ENABLE_DASHBOARD_UPDATES,
        "audit_logging": settings.ENABLE_AUDIT_LOGGING,
    }


def is_production() -> bool:
    """Check if running in production environment"""
    return settings.ENVIRONMENT.lower() == "production"


def is_development() -> bool:
    """Check if running in development environment"""
    return settings.ENVIRONMENT.lower() == "development"


def is_testing() -> bool:
    """Check if running in testing environment"""
    return settings.ENVIRONMENT.lower() in ["test", "testing"]

# Environment-specific configurations


def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration based on environment"""
    base_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            },
            "json": {
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
                "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
            }
        },
        "handlers": {
            "default": {
                "level": settings.LOG_LEVEL,
                "formatter": "json" if settings.LOG_FORMAT == "json" else "standard",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout"
            }
        },
        "loggers": {
            "": {
                "handlers": ["default"],
                "level": settings.LOG_LEVEL,
                "propagate": False
            }
        }
    }

    # Add file handler for production
    if is_production():
        base_config["handlers"]["file"] = {
            "level": settings.LOG_LEVEL,
            "formatter": "json",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
        base_config["loggers"][""]["handlers"].append("file")

    return base_config


def validate_settings() -> None:
    """Validate critical settings"""
    if is_production():
        if settings.SECRET_KEY == "dev-secret-key-change-in-production":
            raise ValueError("SECRET_KEY must be changed in production")

        if not settings.GOOGLE_CLOUD_PROJECT:
            raise ValueError("GOOGLE_CLOUD_PROJECT is required in production")

        if settings.DEBUG:
            raise ValueError("DEBUG must be False in production")


# Export commonly used settings
__all__ = [
    'settings',
    'get_database_config',
    'get_vector_db_config',
    'get_agent_config',
    'get_cache_config',
    'get_security_config',
    'get_monitoring_config',
    'get_feature_flags',
    'get_logging_config',
    'is_production',
    'is_development',
    'is_testing',
    'validate_settings'
]
