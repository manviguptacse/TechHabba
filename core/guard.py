"""Module: core/guard.py
Role: Member 3 - VAKSHA
Purpose: Classifies prompt safety using an injectable classifier adapter
         provided by Member 1. Returns normalized labels: 'safe', 'unsafe', or 'unknown'.
"""

from typing import Any, Callable, Dict, Optional, Union
import json
import time


class GuardError(Exception):
    """Raised when the guard module cannot proceed (e.g. missing adapter)."""
    pass


# Valid classification labels permitted by VAKSHA
VALID_LABELS = {"safe", "unsafe", "unknown"}

# Type alias for classifier adapter: accepts text (str), returns dict, str (JSON/label)
ClassifierAdapter = Callable[[str], Union[str, Dict[str, Any]]]

# Registered default classifier adapter (to be connected by Member 1)
_default_classifier_adapter: Optional[ClassifierAdapter] = None


def set_classifier_adapter(adapter: ClassifierAdapter) -> None:
    """Connect Member 1's actual safety classifier function as the default adapter.

    :param adapter: A callable accepting text (str) and returning str or dict.
    """
    global _default_classifier_adapter
    _default_classifier_adapter = adapter


def get_classifier_adapter() -> Optional[ClassifierAdapter]:
    """Retrieve the currently registered default classifier adapter."""
    return _default_classifier_adapter


def classify_prompt(
    text: str,
    adapter: Optional[ClassifierAdapter] = None,
) -> Dict[str, Any]:
    """Classifies a prompt as 'safe', 'unsafe', or 'unknown' using an injectable adapter.

    :param text: Input prompt to evaluate.
    :param adapter: Optional classifier adapter to override the default.
    :return: A dictionary containing:
             - 'label': 'safe', 'unsafe', or 'unknown'.
             - 'category': Specific safety category/violation tag (if provided, else None).
             - 'model': Identifier/name of the classifier model.
             - 'latency_ms': Execution latency in milliseconds.
             - 'error': Error description if an error occurred (optional).
    :raises GuardError: If no adapter is configured (live integration pending).
    """
    # 1. Resolve classifier adapter
    active_adapter = adapter if adapter is not None else _default_classifier_adapter

    if active_adapter is None:
        raise GuardError(
            "Live integration pending: Safety classifier adapter is not configured. "
            "Member 1's classifier model function must be connected via the adapter."
        )

    model_name = getattr(active_adapter, "__name__", "external_classifier_model")

    # 2. Validate input text
    if not isinstance(text, str) or not text.strip():
        return {
            "label": "unknown",
            "category": None,
            "model": model_name,
            "latency_ms": 0.0,
            "error": "Input text must be a non-empty string.",
        }

    cleaned_text = text.strip()

    # 3. Call adapter and measure latency
    start_time = time.perf_counter()
    raw_response = None
    call_error: Optional[str] = None

    try:
        raw_response = active_adapter(cleaned_text)
    except TimeoutError as exc:
        call_error = f"Classifier request timed out: {exc}"
    except Exception as exc:
        call_error = f"Classifier API error: {exc}"

    end_time = time.perf_counter()
    latency_ms = round((end_time - start_time) * 1000, 2)

    # 4. Handle execution errors (never convert errors into 'safe')
    if call_error is not None:
        return {
            "label": "unknown",
            "category": None,
            "model": model_name,
            "latency_ms": latency_ms,
            "error": call_error,
        }

    # 5. Parse response format (supports dict or JSON/plain string)
    parsed_data: Dict[str, Any] = {}
    if isinstance(raw_response, dict):
        parsed_data = raw_response
    elif isinstance(raw_response, str):
        raw_str = raw_response.strip()
        try:
            parsed_json = json.loads(raw_str)
            if isinstance(parsed_json, dict):
                parsed_data = parsed_json
            else:
                parsed_data = {"label": str(parsed_json)}
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

    # 6. Extract label and category
    raw_label = parsed_data.get("label") or parsed_data.get("status")
    category = parsed_data.get("category")
    if category is not None:
        category = str(category).strip() or None

    detected_model = parsed_data.get("model", model_name)

    # 7. Normalize label and enforce allowed values
    normalized_label = "unknown"
    if raw_label is not None:
        candidate_label = str(raw_label).strip().lower()
        if candidate_label in VALID_LABELS:
            normalized_label = candidate_label

    result: Dict[str, Any] = {
        "label": normalized_label,
        "category": category,
        "model": str(detected_model),
        "latency_ms": latency_ms,
    }

    # If parsing returned an invalid or missing label, record reason
    if normalized_label == "unknown" and raw_label != "unknown":
        result["error"] = f"Invalid or ambiguous classifier label: {raw_label!r}"

    return result
