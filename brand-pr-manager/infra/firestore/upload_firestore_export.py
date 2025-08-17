#!/usr/bin/env python3
"""Upload a Firestore export JSON into Firestore under the `Company` collection.

This script reads an export JSON (format similar to the attached
`firestore_export_20250817_121843.json`) and writes each document and
its nested subcollections into the live Firestore database.

By default the script performs a dry-run and only prints the operations.
Pass --commit to actually perform writes using the Admin SDK.

Usage:
  python upload_firestore_export.py --file /path/to/firestore_export.json [--project PROJECT_ID] [--commit]

Environment:
  Set GOOGLE_APPLICATION_CREDENTIALS to point to a service account JSON with Firestore Admin permissions,
  or rely on the environment where application default credentials are available.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict


def load_export(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_document(db, full_path: str, doc_id: str, data: Dict[str, Any], commit: bool) -> None:
    doc_ref = db.document(f"{full_path}/{doc_id}")
    if commit:
        doc_ref.set(data)
        print(f"WROTE: {full_path}/{doc_id}")
    else:
        print(f"DRYRUN: would write document {full_path}/{doc_id} with keys: {list(data.keys())}")


def _process_subcollections(db, base_path: str, doc_id: str, subcollections: Dict[str, Any], commit: bool) -> None:
    for subcol_name, subcol_obj in (subcollections or {}).items():
        documents = subcol_obj.get("documents", {})
        for subdoc_id, subdoc_obj in documents.items():
            subdoc_data = subdoc_obj.get("data", {})
            full_sub_path = f"{base_path}/{doc_id}/{subcol_name}"
            _write_document(db, full_sub_path, subdoc_id, subdoc_data, commit)

            # Recurse deeper if this subdoc has its own subcollections
            nested_subcols = subdoc_obj.get("subcollections", {})
            if nested_subcols:
                _process_subcollections(db, f"{base_path}/{doc_id}/{subcol_name}", subdoc_id, nested_subcols, commit)


def import_company_collection(export: Dict[str, Any], db, commit: bool, collection_name: str = "Company") -> None:
    data = export.get("data", {})
    companies = data.get(collection_name, {}).get("documents", {})
    if not companies:
        print(f"No documents found in export under '{collection_name}'. Nothing to do.")
        return

    for doc_id, doc_obj in companies.items():
        doc_data = doc_obj.get("data", {})
        # Write the top-level Company document
        _write_document(db, collection_name, doc_id, doc_data, commit)

        # Process subcollections recursively
        subcols = doc_obj.get("subcollections", {})
        if subcols:
            _process_subcollections(db, collection_name, doc_id, subcols, commit)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Upload Firestore export JSON into Company collection")
    parser.add_argument("--file", "-f", required=True, help="Path to export JSON file")
    parser.add_argument("--project", "-p", required=False, help="GCP project id (optional if credentials expose project)")
    parser.add_argument("--commit", action="store_true", help="Actually write to Firestore. Without this flag script runs in dry-run mode")
    parser.add_argument("--collection", default="Company", help="Top-level collection name in export (default: Company)")
    args = parser.parse_args(argv)

    export_path = args.file
    if not os.path.exists(export_path):
        print(f"Export file not found: {export_path}")
        sys.exit(2)

    export = load_export(export_path)

    if not args.commit:
        print("Running in DRY-RUN mode. To perform writes pass --commit and ensure GOOGLE_APPLICATION_CREDENTIALS is set.")

    # Lazy import of firebase admin so dry-run can run without the package installed
    if args.commit:
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore
        except Exception as e:
            print("Failed to import firebase_admin. Install with: pip install firebase-admin")
            raise

        # Initialize app
        cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not cred_path:
            print("Environment variable GOOGLE_APPLICATION_CREDENTIALS is not set. Aborting.")
            sys.exit(2)

        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {'projectId': args.project} if args.project else None)
        db = firestore.client()
    else:
        # Create a lightweight fake client for dry-run printing with same interface used in helpers
        class _DryClient:
            def document(self, path: str):
                class _DocRef:
                    def __init__(self, p):
                        self._p = p

                    def set(self, data):
                        print(f"DRYRUN: would set document {self._p} with keys: {list(data.keys())}")

                return _DocRef(path)

        db = _DryClient()

    import_company_collection(export, db, commit=args.commit, collection_name=args.collection)


if __name__ == "__main__":
    main()
