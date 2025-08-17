#!/usr/bin/env python3
"""
Export all Firestore collections, subcollections, and documents recursively.
Captures the complete data hierarchy including nested collections.
"""

import firebase_admin
from firebase_admin import credentials, firestore
import json
from typing import Any, Dict, List
from datetime import datetime
import os
from pathlib import Path


def serialize_firestore_data(obj: Any) -> Any:
    """Convert Firestore special types to JSON-serializable format."""
    if hasattr(obj, 'isoformat'):  # Handle any datetime-like object
        return obj.isoformat()
    elif hasattr(obj, 'path'):  # Handle document references
        return str(obj.path)
    elif isinstance(obj, dict):
        return {k: serialize_firestore_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_firestore_data(v) for v in obj]
    return obj


class FirestoreExporter:
    def __init__(self, credential_path: str = "sk.json"):
        """Initialize Firebase with credentials."""
        try:
            cred = credentials.Certificate(credential_path)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            print("Connected to Firestore successfully")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Firestore: {e}")

    def extract_schema(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Extract schema from a document."""
        schema = {}
        for key, value in data.items():
            if isinstance(value, dict):
                schema[key] = self.extract_schema(value)
            elif isinstance(value, list):
                if value:
                    if isinstance(value[0], dict):
                        schema[key] = f"Array[{self.extract_schema(value[0])}]"
                    else:
                        schema[key] = f"Array[{type(value[0]).__name__}]"
                else:
                    schema[key] = "Array"
            else:
                schema[key] = type(value).__name__
        return schema

    def get_all_subcollections(self, doc_ref) -> List[str]:
        """Get all subcollections for a document."""
        return [col.id for col in doc_ref.collections()]

    def export_document(self, doc_ref) -> Dict[str, Any]:
        """Export a document and all its subcollections recursively."""
        doc_snapshot = doc_ref.get()
        if not doc_snapshot.exists:
            return None

        doc_data = doc_snapshot.to_dict()
        result = {
            "data": serialize_firestore_data(doc_data),
            "subcollections": {}
        }

        # Get all subcollections
        subcollections = self.get_all_subcollections(doc_ref)
        for subcol in subcollections:
            subcol_ref = doc_ref.collection(subcol)
            result["subcollections"][subcol] = self.export_collection(
                subcol_ref)

        return result

    def export_collection(self, collection_ref) -> Dict[str, Any]:
        """Export a collection and all its documents recursively."""
        result = {
            "documents": {},
            "schema": {}
        }

        docs = collection_ref.get()
        for doc in docs:
            exported_doc = self.export_document(doc.reference)
            if exported_doc:
                result["documents"][doc.id] = exported_doc
                # Extract schema from first document
                if not result["schema"] and exported_doc["data"]:
                    result["schema"] = self.extract_schema(
                        exported_doc["data"])

        return result

    def export_all(self) -> Dict[str, Any]:
        """Export all collections and their contents."""
        try:
            collections = self.db.collections()

            export_data = {
                "metadata": {
                    "exported_at": datetime.now().isoformat(),
                    "collections": {}
                },
                "data": {}
            }

            for col in collections:
                print(f"\nExporting collection: {col.id}")
                collection_data = self.export_collection(col)
                export_data["data"][col.id] = collection_data
                # Add collection metadata
                export_data["metadata"]["collections"][col.id] = {
                    "document_count": len(collection_data["documents"]),
                    "has_schema": bool(collection_data["schema"])
                }

            return export_data
        except Exception as e:
            raise Exception(f"Failed to export collections: {e}")

    def save_to_json(self, data: Dict[str, Any], filename: str = None) -> str:
        """Save data to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"firestore_export_{timestamp}.json"

        try:
            filepath = Path(filename)
            with filepath.open('w') as f:
                json.dump(serialize_firestore_data(data), f, indent=2)
            print(f"\nData saved to: {filepath.absolute()}")
            return str(filepath.absolute())
        except Exception as e:
            raise Exception(f"Failed to save JSON file: {e}")


def main():
    """Main execution function."""
    try:
        # Initialize FirestoreExporter
        exporter = FirestoreExporter()

        # Export all collections
        print("\nStarting full Firestore export...")
        export_data = exporter.export_all()

        # Save to JSON
        json_path = exporter.save_to_json(export_data)

        print("\nExport Summary:")
        print("==============")
        for col_name, col_meta in export_data["metadata"]["collections"].items():
            print(f"\nCollection: {col_name}")
            print(f"Documents: {col_meta['document_count']}")
            if col_meta['has_schema']:
                print("Schema:")
                print(json.dumps(export_data["data"]
                      [col_name]["schema"], indent=2))

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
