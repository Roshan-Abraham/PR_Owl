#!/usr/bin/env python3
"""Read and display the schema of Firestore collections."""

import firebase_admin
from firebase_admin import credentials, firestore
import json
from typing import Any, Dict


def print_schema(data: Dict[str, Any], indent: int = 0) -> Dict[str, str]:
    """Convert a Firestore document into a schema representation."""
    schema = {}
    for key, value in data.items():
        if isinstance(value, dict):
            schema[key] = print_schema(value, indent + 2)
        elif isinstance(value, list):
            if value:
                schema[key] = f"Array[{type(value[0]).__name__}]"
            else:
                schema[key] = "Array"
        else:
            schema[key] = type(value).__name__
    return schema


def main():
    # Initialize Firebase (assuming service account JSON is in the same directory)
    try:
        cred = credentials.Certificate("sk.json")
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Connected to Firestore successfully")
    except Exception as e:
        print(f"Error connecting to Firestore: {e}")
        return

    # Query the companies collection
    try:
        print("\nListing all collections:")
        print("========================")
        collections = db.collections()
        for collection in collections:
            print(f"Found collection: {collection.id}")
        
        print("\nReading Company collection:")
        print("========================")
        company_collection = db.collection('Company')
        all_docs = company_collection.get()
        
        if not all_docs:
            print("No documents found in Company collection")
            return
            
        # Get the first document to extract schema
        for doc in all_docs:
            data = doc.to_dict()
            schema = print_schema(data)
            
            # Save schema to a JSON file
            output_file = 'company_schema.json'
            with open(output_file, 'w') as f:
                json.dump(schema, f, indent=2)
            
            print("\nSchema found:")
            print(json.dumps(schema, indent=2))
            print(f"\nSchema saved to: {output_file}")
            break  # We only need one document to get the schema
            
        # We'll just use the first document to get the schema
        # since all companies should follow the same schema


    except Exception as e:
        print(f"Error reading from Firestore: {e}")


if __name__ == "__main__":
    main()
