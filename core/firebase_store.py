"""Firestore persistence helper for VAKSHA prompt logs and evaluations."""

from datetime import datetime, timezone
import os
from typing import Any, Dict, List, Optional

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except ImportError:  # Allows running UI/tests even if firebase-admin isn't installed yet
    firebase_admin = None  # type: ignore[assignment]
    credentials = None  # type: ignore[assignment]
    firestore = None  # type: ignore[assignment]


def _get_firestore_client():
    if firebase_admin is None or firestore is None:
        return None, "firebase-admin is not installed."

    try:
        if not firebase_admin._apps:
            cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "").strip()
            project_id = os.getenv("FIREBASE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
            options = {"projectId": project_id} if project_id else {}

            if cred_path and os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred, options=options or None)
            else:
                firebase_admin.initialize_app(options=options or None)

        return firestore.client(), None
    except Exception as exc:
        return None, f"Firestore initialization failed: {exc}"


def save_run_log(record: Dict[str, Any], collection: str = "vaksha_runs") -> tuple[bool, Optional[str]]:
    client, err = _get_firestore_client()
    if client is None:
        return False, err

    try:
        payload = {
            **record,
            "created_at": datetime.now(timezone.utc),
        }
        client.collection(collection).add(payload)
        return True, None
    except Exception as exc:
        return False, f"Failed to write to Firestore: {exc}"


def fetch_recent_runs(limit: int = 20, collection: str = "vaksha_runs") -> tuple[List[Dict[str, Any]], Optional[str]]:
    client, err = _get_firestore_client()
    if client is None:
        return [], err

    try:
        docs = (
            client.collection(collection)
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        rows: List[Dict[str, Any]] = []
        for doc in docs:
            data = doc.to_dict() or {}
            data["id"] = doc.id
            created = data.get("created_at")
            if hasattr(created, "isoformat"):
                data["created_at"] = created.isoformat()
            rows.append(data)
        return rows, None
    except Exception as exc:
        return [], f"Failed to read from Firestore: {exc}"