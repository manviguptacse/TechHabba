"""Module: core/pipeline.py
Role: Member 3 - VAKSHA
Purpose: Coordinates translation, guardrails, and target model execution
         for both baseline and protected inference routes.
"""

from typing import Any, Callable, Dict, List, Optional, Union
import time

from core.guard import GuardError, classify_prompt
from core.translate import TranslationError, translate_to_english

TargetAdapter = Callable[[str], Union[str, Dict[str, Any]]]
_default_target_adapter: Optional[TargetAdapter] = None


def set_target_adapter(adapter: TargetAdapter) -> None:
    global _default_target_adapter
    _default_target_adapter = adapter


def get_target_adapter() -> Optional[TargetAdapter]:
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


def _invoke_target(
    prompt: str,
    target_adapter: TargetAdapter,
    errors: List[str],
) -> tuple[bool, Optional[str], Optional[Any], str]:
    """Helper that calls the target model with the original prompt and reports accuracy."""
    model_name = getattr(target_adapter, "__name__", "external_target_model")
    try:
        raw_output = target_adapter(prompt)
        if isinstance(raw_output, dict):
            response = (
                raw_output.get("response")
                if raw_output.get("response") is not None
                else raw_output.get("output")
                if raw_output.get("output") is not None
                else raw_output.get("text")
                if raw_output.get("text") is not None
                else str(raw_output)
            )
            model_name = str(raw_output.get("model") or model_name)
        else:
            response = str(raw_output)
        return True, model_name, response, "success"
    except Exception as exc:
        errors.append(f"Target model execution failed: {exc}")
        return False, None, None, "failed"


def run_baseline(
    prompt: str,
    prompt_id: Optional[Any] = None,
    target_adapter: Optional[TargetAdapter] = None,
) -> Dict[str, Any]:
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

    target_called, target_model_name, target_response, status = _invoke_target(
        prompt, active_target, errors
    )
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return _create_result(
        prompt_id=prompt_id,
        original_prompt=prompt,
        route="baseline",
        status=status,
        decision="ALLOW",
        translation=None,
        guard=None,
        target_model=target_model_name,
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

    # Step 1: Translate to English
    try:
        translation_info = translate_to_english(prompt, adapter=translation_adapter)
        english_text = (translation_info.get("translated_text") or "").strip()
        if not english_text:
            raise TranslationError("Translation returned empty text.")
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

    if guard_info.get("error"):
        errors.append(guard_info["error"])

    label = guard_info.get("label", "unknown")

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

    target_called, target_model_name, target_response, status = _invoke_target(
        prompt, active_target, errors
    )
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return _create_result(
        prompt_id=prompt_id,
        original_prompt=prompt,
        route="protected",
        status=status,
        decision="ALLOW",
        translation=translation_info,
        guard=guard_info,
        target_model=target_model_name,
        target_response=target_response,
        target_called=target_called,
        latency_ms=latency_ms,
        errors=errors,
    )