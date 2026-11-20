"""Module: core/pipeline.py
Role: Member 3 - VAKSHA
Purpose: Coordinates translation, guardrails, and target model execution
         for both baseline and protected inference routes.
"""

from typing import Any, Callable, Dict, List, Optional, Union
import time

from core.guard import GuardError, classify_prompt
from core.translate import TranslationError, translate_to_english

# Type alias for the target AI model adapter: accepts text (str), returns str or dict
TargetAdapter = Callable[[str], Union[str, Dict[str, Any]]]

# Registered default target adapter (to be connected by Member 1)
_default_target_adapter: Optional[TargetAdapter] = None


def set_target_adapter(adapter: TargetAdapter) -> None:
    """Connect Member 1's actual target AI model function as the default adapter."""
    global _default_target_adapter
    _default_target_adapter = adapter


def get_target_adapter() -> Optional[TargetAdapter]:
    """Retrieve the currently registered default target AI model adapter."""
    return _default_target_adapter


def _create_result(
    prompt_id: Optional[Any],
    original_prompt: str,
    route: str,
    status: str,
    decision: str,
    translation: Optional[Dict[str, Any]],
    guard: Optional[Dict[str, Any]],
    target_model: Optional[str],
    target_response: Optional[Any],
    target_called: bool,
    latency_ms: float,
    errors: List[str],
) -> Dict[str, Any]:
    """Helper to return a standardized, consistent pipeline result dictionary."""
    return {
        "prompt_id": prompt_id,
        "original_prompt": original_prompt,
        "route": route,
        "status": status,
        "decision": decision,
        "translation": translation,
        "guard": guard,
        "target_model": target_model,
        "target_response": target_response,
        "target_called": target_called,
        "latency_ms": latency_ms,
        "errors": errors,
    }


def run_baseline(
    prompt: str,
    prompt_id: Optional[Any] = None,
    target_adapter: Optional[TargetAdapter] = None,
) -> Dict[str, Any]:
    """Executes the baseline route: directly forwards the original prompt to the target model.

    :param prompt: Original input prompt.
    :param prompt_id: Identifier for tracking prompts.
    :param target_adapter: Optional target model adapter callable.
    :return: Standard pipeline result dictionary.
    """
    start_time = time.perf_counter()
    errors: List[str] = []

    if not isinstance(prompt, str) or not prompt.strip():
        errors.append("Prompt must be a non-empty string.")
        return _create_result(
            prompt_id=prompt_id,
            original_prompt=prompt if isinstance(prompt, str) else "",
            route="baseline",
            status="failed",
            decision="ALLOW",
            translation=None,
            guard=None,
            target_model=None,
            target_response=None,
            target_called=False,
            latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
            errors=errors,
        )

    active_target = target_adapter if target_adapter is not None else _default_target_adapter

    if active_target is None:
        errors.append("Target model adapter is not configured (live integration pending).")
        return _create_result(
            prompt_id=prompt_id,
            original_prompt=prompt,
            route="baseline",
            status="failed",
            decision="ALLOW",
            translation=None,
            guard=None,
            target_model=None,
            target_response=None,
            target_called=False,
            latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
            errors=errors,
        )

    target_model_name = getattr(active_target, "__name__", "external_target_model")
    target_called = True
    target_response: Optional[Any] = None
    status = "success"

    try:
        raw_output = active_target(prompt)
        if isinstance(raw_output, dict):
            target_response = (
                raw_output.get("response")
                or raw_output.get("output")
                or raw_output.get("text")
                or str(raw_output)
            )
            target_model_name = raw_output.get("model", target_model_name)
        else:
            target_response = str(raw_output)
    except Exception as exc:
        errors.append(f"Target model execution failed: {exc}")
        status = "failed"
        target_response = None

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return _create_result(
        prompt_id=prompt_id,
        original_prompt=prompt,
        route="baseline",
        status=status,
        decision="ALLOW",
        translation=None,
        guard=None,
        target_model=target_model_name if target_called else None,
        target_response=target_response,
        target_called=target_called,
        latency_ms=latency_ms,
        errors=errors,
    )


def run_protected(
    prompt: str,
    prompt_id: Optional[Any] = None,
    translation_adapter: Optional[Any] = None,
    classifier_adapter: Optional[Any] = None,
    target_adapter: Optional[TargetAdapter] = None,
) -> Dict[str, Any]:
    """Executes the protected route:
    1. Translates original prompt to English.
    2. Classifies translated text for safety.
    3. If safe: sends ORIGINAL prompt to target model.
    4. If unsafe: blocks without calling target model.
    5. If unknown/errors: routes to review_required without calling target model.

    :param prompt: Original input prompt.
    :param prompt_id: Identifier for tracking prompts.
    :param translation_adapter: Optional translation adapter.
    :param classifier_adapter: Optional classifier adapter.
    :param target_adapter: Optional target model adapter.
    :return: Standard pipeline result dictionary.
    """
    start_time = time.perf_counter()
    errors: List[str] = []

    if not isinstance(prompt, str) or not prompt.strip():
        errors.append("Prompt must be a non-empty string.")
        return _create_result(
            prompt_id=prompt_id,
            original_prompt=prompt if isinstance(prompt, str) else "",
            route="protected",
            status="failed",
            decision="BLOCK",
            translation=None,
            guard=None,
            target_model=None,
            target_response=None,
            target_called=False,
            latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
            errors=errors,
        )

    # Step 1: Translate prompt to English
    translation_info: Optional[Dict[str, Any]] = None
    try:
        translation_info = translate_to_english(prompt, adapter=translation_adapter)
        english_text = translation_info.get("translated_text", "")
    except TranslationError as exc:
        errors.append(f"Translation failed: {exc}")
        return _create_result(
            prompt_id=prompt_id,
            original_prompt=prompt,
            route="protected",
            status="review_required",
            decision="REVIEW_REQUIRED",
            translation=None,
            guard=None,
            target_model=None,
            target_response=None,
            target_called=False,
            latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
            errors=errors,
        )
    except Exception as exc:
        errors.append(f"Unexpected translation error: {exc}")
        return _create_result(
            prompt_id=prompt_id,
            original_prompt=prompt,
            route="protected",
            status="review_required",
            decision="REVIEW_REQUIRED",
            translation=None,
            guard=None,
            target_model=None,
            target_response=None,
            target_called=False,
            latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
            errors=errors,
        )

    # Step 2: Classify translated text
    guard_info: Optional[Dict[str, Any]] = None
    try:
        guard_info = classify_prompt(english_text, adapter=classifier_adapter)
    except GuardError as exc:
        errors.append(f"Classifier failed: {exc}")
        return _create_result(
            prompt_id=prompt_id,
            original_prompt=prompt,
            route="protected",
            status="review_required",
            decision="REVIEW_REQUIRED",
            translation=translation_info,
            guard=None,
            target_model=None,
            target_response=None,
            target_called=False,
            latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
            errors=errors,
        )
    except Exception as exc:
        errors.append(f"Unexpected classifier error: {exc}")
        return _create_result(
            prompt_id=prompt_id,
            original_prompt=prompt,
            route="protected",
            status="review_required",
            decision="REVIEW_REQUIRED",
            translation=translation_info,
            guard=None,
            target_model=None,
            target_response=None,
            target_called=False,
            latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
            errors=errors,
        )

    # Collect guard error notices if present
    if guard_info.get("error"):
        errors.append(guard_info["error"])

    label = guard_info.get("label", "unknown")

    # Step 3: Handle Unsafe Prompts (BLOCK)
    if label == "unsafe":
        return _create_result(
            prompt_id=prompt_id,
            original_prompt=prompt,
            route="protected",
            status="blocked",
            decision="BLOCK",
            translation=translation_info,
            guard=guard_info,
            target_model=None,
            target_response=None,
            target_called=False,
            latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
            errors=errors,
        )

    # Step 4: Handle Ambiguous or Unknown Prompts (REVIEW_REQUIRED)
    if label != "safe":
        return _create_result(
            prompt_id=prompt_id,
            original_prompt=prompt,
            route="protected",
            status="review_required",
            decision="REVIEW_REQUIRED",
            translation=translation_info,
            guard=guard_info,
            target_model=None,
            target_response=None,
            target_called=False,
            latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
            errors=errors,
        )

    # Step 5: Handle Safe Prompts (ALLOW) -> Call Target Model with ORIGINAL prompt
    active_target = target_adapter if target_adapter is not None else _default_target_adapter

    if active_target is None:
        errors.append("Target model adapter is not configured (live integration pending).")
        return _create_result(
            prompt_id=prompt_id,
            original_prompt=prompt,
            route="protected",
            status="failed",
            decision="ALLOW",
            translation=translation_info,
            guard=guard_info,
            target_model=None,
            target_response=None,
            target_called=False,
            latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
            errors=errors,
        )

    target_model_name = getattr(active_target, "__name__", "external_target_model")
    target_called = True
    target_response = None
    status = "success"

    try:
        # NOTE: Send original prompt to target model, NEVER the translated text
        raw_target_resp = active_target(prompt)
        if isinstance(raw_target_resp, dict):
            target_response = (
                raw_target_resp.get("response")
                or raw_target_resp.get("output")
                or raw_target_resp.get("text")
                or str(raw_target_resp)
            )
            target_model_name = raw_target_resp.get("model", target_model_name)
        else:
            target_response = str(raw_target_resp)
    except Exception as exc:
        errors.append(f"Target model execution failed: {exc}")
        status = "failed"
        target_response = None

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return _create_result(
        prompt_id=prompt_id,
        original_prompt=prompt,
        route="protected",
        status=status,
        decision="ALLOW",
        translation=translation_info,
        guard=guard_info,
        target_model=target_model_name if target_called else None,
        target_response=target_response,
        target_called=target_called,
        latency_ms=latency_ms,
        errors=errors,
    )
