#!/usr/bin/env python3
"""Seed Firestore with dashboard documents for development/testing.

Usage:
  export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
  python3 seed_firestore.py --project <PROJECT_ID> --company company_acme
"""
import argparse
import json
from datetime import datetime, timezone

import firebase_admin
from firebase_admin import credentials, firestore


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def initialize_firestore(project_id, service_account_file):
    """Initializes Firestore client using a service account."""
    try:
        cred = credentials.Certificate(service_account_file)
        firebase_admin.initialize_app(cred, {
            'projectId': project_id,
        })
        print("Firebase app initialized successfully.")
        return firestore.client()
    except Exception as e:
        print(f"Error initializing Firebase app: {e}")
        print("Please ensure you have a valid service account key file and the path is correct.")
        exit(1)


def create_company_dashboard(db, company_id, name):
    doc_ref = db.document(f"dashboards/companies/{company_id}/summary")
    payload = {
        "meta": {"company_id": company_id, "name": name, "last_updated": now_iso()},
        "totals": {
            "num_crises": 0,
            "num_active_crises": 0,
            "num_resolved_24h": 0,
            "num_classifications": 0,
            "num_recommendations": 0,
            "num_artifacts": 0,
            "num_agent_runs": 0,
            "num_vector_objects": 0
        },
        "rates": {"crises_per_7d": 0, "classifications_per_7d": 0},
        "quality": {"avg_classification_confidence": 0.0, "avg_severity_score": 0.0, "classification_confidence_histogram": {}},
        "distributions": {"class_distribution": {}},
        "performance": {"avg_agent_run_time_ms": 0, "avg_tokens_per_run": 0},
        "alerts": {"num_recent_errors_24h": 0, "top_error_codes": {}}
    }
    doc_ref.set(payload)
    print(f"Wrote dashboards/companies/{company_id}/summary")


def create_crisis_stats(db, company_id, crisis_id):
    doc_ref = db.document(f"dashboards/crises/{company_id}/{crisis_id}")
    payload = {
        "meta": {"crisis_id": crisis_id, "company_id": company_id, "created_at": now_iso(), "origin_point": {"type": "simulation"}},
        "summary": {"current_status": "created", "severity_score_latest": 0.0, "confidence_score_latest": 0.0, "num_artifacts": 0, "num_classifications": 0, "num_recommendations": 0, "num_logs": 0},
        "timeline": {"last_classification_at": None, "last_recommendation_at": None, "last_snapshot_at": None},
        "top_entities": [],
        "owners": {"assigned_team": None, "owner_user_id": None}
    }
    doc_ref.set(payload)
    print(f"Wrote dashboards/crises/{company_id}/crisis_stats/{crisis_id}")


def create_agents_summary(db, company_id):
    doc_ref = db.document(f"dashboards/agents/{company_id}/summary")
    payload = {
        "totals": {"num_agent_runs": 0, "num_failed_runs": 0, "num_success_runs": 0},
        "latency": {"avg_run_time_ms": 0, "p95_run_time_ms": 0},
        "cost": {"tokens_consumed_total": 0, "tokens_per_day": 0},
        "model_usage": {},
        "errors": {},
        "top_slow_agents": []
    }
    doc_ref.set(payload)
    print(f"Wrote dashboards/agents/{company_id}/summary")


def create_vectors_summary(db, company_id):
    doc_ref = db.document(f"dashboards/vectors/{company_id}/summary")
    payload = {
        "num_vectors_total": 0,
        "num_vectors_created_7d": 0,
        "last_ingest_at": None,
        "index_health": {},
        "avg_vector_age_days": 0.0
    }
    doc_ref.set(payload)
    print(f"Wrote dashboards/vectors/{company_id}/summary")


def create_system_dashboard(db, env="dev"):
    doc_ref = db.document(f"dashboards/system/{env}/stats")
    payload = {
        "meta": {"env": env, "last_updated": now_iso()},
        "totals": {"total_companies": 0, "total_crises": 0, "total_agent_runs": 0, "total_errors_24h": 0},
        "capacity": {"firestore_ops_per_minute": 0, "milvus_query_rate": 0},
        "incidents": [],
        "backups": {"last_backup_at": None, "last_backup_status": None}
    }
    doc_ref.set(payload)
    print(f"Wrote dashboards/system/{env}/stats")


def main():
    # Hardcoded values
    project_id = "prowl-24f3e"
    service_account_file = "/workspaces/context-engineering-intro/sk.json"
    company_id = "FakeHub"
    env = "dev"

    db = initialize_firestore(project_id, service_account_file)

    print(f"Seeding data for company: {company_id}")
    create_company_dashboard(db, company_id, "ACME Inc.")
    create_crisis_stats(db, company_id, "crisis_123")
    create_agents_summary(db, company_id)
    create_vectors_summary(db, company_id)
    create_system_dashboard(db, env)
    print("Done.")


if __name__ == "__main__":
    main()
