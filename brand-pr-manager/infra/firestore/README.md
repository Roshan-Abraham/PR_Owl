This folder contains Firestore deployment artifacts and a simple seed script for the dashboard documents.

Quick steps to deploy and seed Firestore

1. Install Firebase CLI and authenticate:
   - npm install -g firebase-tools
   - firebase login

2. Set your project alias (or use project id directly):
   - firebase use --add <PROJECT_ID>

3. Deploy Firestore rules and indexes:
   - firebase deploy --only firestore:rules,firestore:indexes

4. Seed example dashboard documents (requires service account credentials):
   - export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
   - python3 seed_firestore.py --project <PROJECT_ID> --company company_acme

Files in this directory:
- firestore.rules          Firestore security rules for client UI (server uses Admin SDK)
- firestore.indexes.json   Composite index definitions for Firestore
- firebase.json            Firebase CLI config pointing to rules/indexes
- seed_firestore.py        Small Python script that writes sample dashboard docs
- requirements.txt         Minimal Python dependencies for seeding script

Notes:
- Server-side agents should use the Admin SDK (service account) which bypasses Firestore rules.
- Use nightly recompute jobs (Cloud Functions / Cloud Scheduler) to reconcile aggregates if you use incremental counters.
- Update `seed_firestore.py` with your project's IDs and any additional fields before running in production.
