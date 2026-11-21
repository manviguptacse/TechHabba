"""Module: core/guard.py
Role: Member 3 - VAKSHA
Purpose: Classifies prompt safety using an injectable classifier adapter.
         Returns normalized labels: 'safe', 'unsafe', or 'unknown'.
"""

from typing import Any, Callable, Dict, Optional, Union
import json
import time


class GuardError(Exception):
    """Raised when the guard module cannot proceed (e.g. missing adapter)."""


VALID_LABELS = {"safe", "unsafe", "unknown"}
ClassifierAdapter = Callable[[str], Union[str, Dict[str, Any]]]

_default_classifier_adapter: Optional[ClassifierAdapter] = None


def set_classifier_adapter(adapter: ClassifierAdapter) -> None:
    global _default_classifier_adapter
    _default_classifier_adapter = adapter


def get_classifier_adapter() -> Optional[ClassifierAdapter]:
    return _default_classifier_adapter


def classify_prompt(
    text: str,
    adapter: Optional[ClassifierAdapter] = None,
) -> Dict[str, Any]:
    active_adapter = adapter if adapter is not None else _default_classifier_adapter

    if active_adapter is None:
        raise GuardError(
            "Live integration pending: Safety classifier adapter is not configured. "
            "Classifier model function must be connected via the adapter."
        )

    model_name = getattr(active_adapter, "__name__", "external_classifier_model")

    if not isinstance(text, str) or not text.strip():
        return {
            "label": "unknown",
            "category": None,
            "model": model_name,
            "latency_ms": 0.0,
            "error": "Input text must be a non-empty string.",
        }

    cleaned_text = text.strip()
    start_time = time.perf_counter()

    try:
        raw_response = active_adapter(cleaned_text)
    except TimeoutError as exc:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "label": "unknown",
            "category": None,
            "model": model_name,
            "latency_ms": latency_ms,
            "error": f"Classifier request timed out: {exc}",
        }
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "label": "unknown",
            "category": None,
            "model": model_name,
            "latency_ms": latency_ms,
            "error": f"Classifier API error: {exc}",
        }

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    parsed_data: Dict[str, Any] = {}
    if isinstance(raw_response, dict):
        parsed_data = raw_response
    elif isinstance(raw_response, str):
        raw_str = raw_response.strip()
        if not raw_str:
            return {
                "label": "unknown",
                "category": None,
                "model": model_name,
                "latency_ms": latency_ms,
                "error": "Classifier returned an empty string.",
            }
        try:
            parsed_json = json.loads(raw_str)
            parsed_data = parsed_json if isinstance(parsed_json, dict) else {"label": str(parsed_json)}
        except json.JSONDecodeError:
            parsed_data = {"label": raw_str}
    else:
        return {
            "label": "unknown",
            "category": None,
            "model": model_name,
            "latency_ms": latency_ms,
            "error": f"Unexpected classifier response type: {type(raw_response).__name__}",
        }

    raw_label = parsed_data.get("label") if "label" in parsed_data else parsed_data.get("status")
    raw_category = parsed_data.get("category")
    category: Optional[str] = None
    if raw_category is not None:
        cat_str = str(raw_category).strip()
        category = cat_str if cat_str else None

    detected_model = str(parsed_data.get("model") or model_name)

    normalized_label = "unknown"
    if raw_label is not None:
        candidate = str(raw_label).strip().lower()
        if candidate in VALID_LABELS:
            normalized_label = candidate

    result: Dict[str, Any] = {
        "label": normalized_label,
        "category": category,
        "model": detected_model,
        "latency_ms": latency_ms,
    }

    if normalized_label == "unknown" and str(raw_label).strip().lower() != "unknown":
        result["error"] = f"Invalid or ambiguous classifier label: {raw_label!r}"

    return result