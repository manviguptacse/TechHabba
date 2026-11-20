"""Tests: tests/test_pipeline.py
Role: Member 3 - VAKSHA
Purpose: Comprehensive unit test suite for VAKSHA core modules and pipeline
         using strictly mocked adapters (no network access, no real API keys).
"""

from typing import Any, Dict, List
import pytest

from core.pipeline import run_baseline, run_protected


# ============================================================================
# Mock Adapter Helpers
# ============================================================================

class MockCallRecorder:
    """Helper to record inputs and control outputs of mock adapters."""

    def __init__(self):
        self.translation_calls: List[str] = []
        self.classifier_calls: List[str] = []
        self.target_calls: List[str] = []

    def make_translation_adapter(
        self,
        translated_text: str = "This is a safe English translation.",
        model_name: str = "mock-translation-model",
    ):
        def adapter(text: str) -> Dict[str, Any]:
            self.translation_calls.append(text)
            return {
                "translated_text": translated_text,
                "model": model_name,
                "latency_ms": 1.5,
            }
        return adapter

    def make_classifier_adapter(
        self,
        label: str = "safe",
        category: Any = None,
        model_name: str = "mock-classifier-model",
    ):
        def adapter(text: str) -> Dict[str, Any]:
            self.classifier_calls.append(text)
            return {
                "label": label,
                "category": category,
                "model": model_name,
            }
        return adapter

    def make_target_adapter(
        self,
        response_text: str = "Standard model response.",
        model_name: str = "mock-target-model",
    ):
        def adapter(text: str) -> Dict[str, Any]:
            self.target_calls.append(text)
            return {
                "response": response_text,
                "model": model_name,
            }
        return adapter


@pytest.fixture
def recorder() -> MockCallRecorder:
    """Provides a fresh CallRecorder instance per test."""
    return MockCallRecorder()


# ============================================================================
# 1. Empty Input Tests
# ============================================================================

def test_empty_input_baseline(recorder: MockCallRecorder):
    """Empty or whitespace-only input in baseline should fail cleanly."""
    for empty_val in ["", "   ", "\t\n"]:
        res = run_baseline(empty_val, prompt_id="p-empty-1", target_adapter=recorder.make_target_adapter())
        assert res["status"] == "failed"
        assert res["target_called"] is False
        assert res["target_response"] is None
        assert len(res["errors"]) > 0
    assert len(recorder.target_calls) == 0


def test_empty_input_protected(recorder: MockCallRecorder):
    """Empty or whitespace-only input in protected route should fail cleanly."""
    for empty_val in ["", "   ", "\t\n"]:
        res = run_protected(
            empty_val,
            prompt_id="p-empty-2",
            translation_adapter=recorder.make_translation_adapter(),
            classifier_adapter=recorder.make_classifier_adapter(),
            target_adapter=recorder.make_target_adapter(),
        )
        assert res["status"] == "failed"
        assert res["decision"] == "BLOCK"
        assert res["target_called"] is False
        assert res["target_response"] is None
        assert len(res["errors"]) > 0
    assert len(recorder.target_calls) == 0


# ============================================================================
# 2. Baseline Sends Original Prompt
# ============================================================================

def test_baseline_sends_original_prompt(recorder: MockCallRecorder):
    """Baseline route must forward the original input prompt directly to the target model."""
    prompt = "ಇದು ನೇರವಾದ ಪರೀಕ್ಷಾ ಪ್ರಾಂಪ್ಟ್"
    target = recorder.make_target_adapter(response_text="Direct baseline reply.")

    res = run_baseline(prompt, prompt_id="base-101", target_adapter=target)

    assert res["status"] == "success"
    assert res["decision"] == "ALLOW"
    assert res["target_called"] is True
    assert res["original_prompt"] == prompt
    assert res["target_response"] == "Direct baseline reply."
    assert recorder.target_calls == [prompt]


# ============================================================================
# 3. Safe Protected Request Reaches Target Model
# ============================================================================

def test_safe_protected_request_reaches_target_model(recorder: MockCallRecorder):
    """When translation succeeds and classifier labels as 'safe', target model is reached."""
    prompt = "ನನ್ನ ಊರು ಬೆಂಗಳೂರು"
    trans = recorder.make_translation_adapter("My hometown is Bengaluru")
    guard = recorder.make_classifier_adapter(label="safe")
    target = recorder.make_target_adapter("Bengaluru is a great city")

    res = run_protected(
        prompt,
        prompt_id="prot-safe-1",
        translation_adapter=trans,
        classifier_adapter=guard,
        target_adapter=target,
    )

    assert res["status"] == "success"
    assert res["decision"] == "ALLOW"
    assert res["target_called"] is True
    assert res["target_response"] == "Bengaluru is a great city"
    assert len(recorder.target_calls) == 1


# ============================================================================
# 4. Safe Request Sends Original Kannada/Kanglish Text, NOT Translation
# ============================================================================

def test_safe_request_sends_original_text_not_translation(recorder: MockCallRecorder):
    """CRITICAL RULE: The target model MUST receive the original text, NEVER the English translation."""
    original_prompt = "ಹೇಗಿದ್ದೀರಾ, ಇವತ್ತಿನ ದಿನ ಹೇಗಿದೆ?"
    translated_english = "How are you, how is today?"

    trans = recorder.make_translation_adapter(translated_text=translated_english)
    guard = recorder.make_classifier_adapter(label="safe")
    target = recorder.make_target_adapter()

    run_protected(
        original_prompt,
        prompt_id="prot-lang-1",
        translation_adapter=trans,
        classifier_adapter=guard,
        target_adapter=target,
    )

    # Verify target model received the original Kannada text
    assert recorder.target_calls == [original_prompt]
    assert translated_english not in recorder.target_calls


# ============================================================================
# 5 & 6. Unsafe Request Is Blocked and Never Calls Target Model
# ============================================================================

def test_unsafe_request_is_blocked_and_never_calls_target(recorder: MockCallRecorder):
    """Unsafe requests must be blocked with decision BLOCK and target model must NOT be called."""
    unsafe_prompt = "ಹಾನಿಕಾರಕ ಪ್ರಾಂಪ್ಟ್"
    trans = recorder.make_translation_adapter("Harmful prompt in English")
    guard = recorder.make_classifier_adapter(label="unsafe", category="hate_speech")
    target = recorder.make_target_adapter()

    res = run_protected(
        unsafe_prompt,
        prompt_id="prot-unsafe-1",
        translation_adapter=trans,
        classifier_adapter=guard,
        target_adapter=target,
    )

    assert res["decision"] == "BLOCK"
    assert res["status"] == "blocked"
    assert res["target_called"] is False
    assert res["target_response"] is None
    assert len(recorder.target_calls) == 0


# ============================================================================
# 7. Unknown Classification Requires Review
# ============================================================================

def test_unknown_classification_requires_review(recorder: MockCallRecorder):
    """When classifier returns unknown, request routes to REVIEW_REQUIRED without calling target."""
    prompt = "ಅಸ್ಪಷ್ಟ ಪ್ರಾಂಪ್ಟ್"
    trans = recorder.make_translation_adapter("Ambiguous statement")
    guard = recorder.make_classifier_adapter(label="unknown")
    target = recorder.make_target_adapter()

    res = run_protected(
        prompt,
        prompt_id="prot-unknown-1",
        translation_adapter=trans,
        classifier_adapter=guard,
        target_adapter=target,
    )

    assert res["decision"] == "REVIEW_REQUIRED"
    assert res["status"] == "review_required"
    assert res["target_called"] is False
    assert res["target_response"] is None
    assert len(recorder.target_calls) == 0


# ============================================================================
# 8. Invalid Classifier Response Becomes Unknown
# ============================================================================

def test_invalid_classifier_response_becomes_unknown(recorder: MockCallRecorder):
    """Classifier returning invalid JSON, unrecognized label, or unexpected format becomes unknown."""
    prompt = "Sample text"
    trans = recorder.make_translation_adapter("Sample translated text")

    # Adapter returning unrecognized label string
    def bad_guard_adapter(text: str):
        return {"label": "UNRECOGNIZED_LABEL_XYZ"}

    target = recorder.make_target_adapter()

    res = run_protected(
        prompt,
        prompt_id="prot-invalid-guard",
        translation_adapter=trans,
        classifier_adapter=bad_guard_adapter,
        target_adapter=target,
    )

    assert res["decision"] == "REVIEW_REQUIRED"
    assert res["status"] == "review_required"
    assert res["guard"]["label"] == "unknown"
    assert res["target_called"] is False
    assert res["target_response"] is None
    assert len(recorder.target_calls) == 0


# ============================================================================
# 9 & 10. Translation Failure & Timeout Require Review
# ============================================================================

def test_translation_failure_requires_review(recorder: MockCallRecorder):
    """A general exception in the translation adapter must route to REVIEW_REQUIRED."""
    def failing_translation(text: str):
        raise RuntimeError("Translation API network disconnect")

    guard = recorder.make_classifier_adapter(label="safe")
    target = recorder.make_target_adapter()

    res = run_protected(
        "ಯಾವುದೋ ಪಠ್ಯ",
        prompt_id="prot-trans-fail",
        translation_adapter=failing_translation,
        classifier_adapter=guard,
        target_adapter=target,
    )

    assert res["decision"] == "REVIEW_REQUIRED"
    assert res["status"] == "review_required"
    assert res["target_called"] is False
    assert res["target_response"] is None
    assert len(recorder.target_calls) == 0
    assert any("Translation failed" in err for err in res["errors"])


def test_translation_timeout_requires_review(recorder: MockCallRecorder):
    """A timeout in the translation adapter must route to REVIEW_REQUIRED."""
    def timeout_translation(text: str):
        raise TimeoutError("Translation service timeout after 5000ms")

    guard = recorder.make_classifier_adapter(label="safe")
    target = recorder.make_target_adapter()

    res = run_protected(
        "ಯಾವುದೋ ಪಠ್ಯ",
        prompt_id="prot-trans-timeout",
        translation_adapter=timeout_translation,
        classifier_adapter=guard,
        target_adapter=target,
    )

    assert res["decision"] == "REVIEW_REQUIRED"
    assert res["status"] == "review_required"
    assert res["target_called"] is False
    assert res["target_response"] is None
    assert any("timed out" in err.lower() for err in res["errors"])


# ============================================================================
# 11 & 12. Classifier Failure & Timeout Require Review
# ============================================================================

def test_classifier_failure_requires_review(recorder: MockCallRecorder):
    """An API failure in the classifier adapter must route to REVIEW_REQUIRED and never safe."""
    trans = recorder.make_translation_adapter("Valid translation")

    def failing_classifier(text: str):
        raise RuntimeError("Classifier internal server error (500)")

    target = recorder.make_target_adapter()

    res = run_protected(
        "ಪಠ್ಯ",
        prompt_id="prot-guard-fail",
        translation_adapter=trans,
        classifier_adapter=failing_classifier,
        target_adapter=target,
    )

    assert res["decision"] == "REVIEW_REQUIRED"
    assert res["status"] == "review_required"
    assert res["target_called"] is False
    assert res["target_response"] is None
    assert len(recorder.target_calls) == 0


def test_classifier_timeout_requires_review(recorder: MockCallRecorder):
    """A timeout in the classifier adapter must route to REVIEW_REQUIRED and never safe."""
    trans = recorder.make_translation_adapter("Valid translation")

    def timeout_classifier(text: str):
        raise TimeoutError("Classifier connection deadline exceeded")

    target = recorder.make_target_adapter()

    res = run_protected(
        "ಪಠ್ಯ",
        prompt_id="prot-guard-timeout",
        translation_adapter=trans,
        classifier_adapter=timeout_classifier,
        target_adapter=target,
    )

    assert res["decision"] == "REVIEW_REQUIRED"
    assert res["status"] == "review_required"
    assert res["target_called"] is False
    assert res["target_response"] is None
    assert len(recorder.target_calls) == 0


# ============================================================================
# 13 & 14. Target Model Failure & Timeout
# ============================================================================

def test_target_model_failure_in_baseline_and_protected(recorder: MockCallRecorder):
    """When target model raises an exception, the status must be recorded as 'failed'."""
    def failing_target(text: str):
        raise RuntimeError("Target model CUDA out of memory")

    # Baseline
    res_base = run_baseline("Prompt", prompt_id="base-fail", target_adapter=failing_target)
    assert res_base["status"] == "failed"
    assert res_base["target_called"] is True
    assert res_base["target_response"] is None
    assert len(res_base["errors"]) > 0

    # Protected
    trans = recorder.make_translation_adapter("Safe translation")
    guard = recorder.make_classifier_adapter(label="safe")
    res_prot = run_protected(
        "Prompt",
        prompt_id="prot-fail",
        translation_adapter=trans,
        classifier_adapter=guard,
        target_adapter=failing_target,
    )
    assert res_prot["status"] == "failed"
    assert res_prot["decision"] == "ALLOW"
    assert res_prot["target_called"] is True
    assert res_prot["target_response"] is None
    assert len(res_prot["errors"]) > 0


def test_target_model_timeout_in_baseline_and_protected(recorder: MockCallRecorder):
    """When target model times out, the status must be recorded as 'failed'."""
    def timeout_target(text: str):
        raise TimeoutError("Target model request timed out")

    # Baseline
    res_base = run_baseline("Prompt", prompt_id="base-timeout", target_adapter=timeout_target)
    assert res_base["status"] == "failed"
    assert res_base["target_called"] is True
    assert res_base["target_response"] is None

    # Protected
    trans = recorder.make_translation_adapter("Safe translation")
    guard = recorder.make_classifier_adapter(label="safe")
    res_prot = run_protected(
        "Prompt",
        prompt_id="prot-timeout",
        translation_adapter=trans,
        classifier_adapter=guard,
        target_adapter=timeout_target,
    )
    assert res_prot["status"] == "failed"
    assert res_prot["decision"] == "ALLOW"
    assert res_prot["target_called"] is True
    assert res_prot["target_response"] is None


# ============================================================================
# 15 & 16. Kannada and Kanglish Inputs Reach Translation Adapter
# ============================================================================

def test_kannada_input_reaches_translation_adapter(recorder: MockCallRecorder):
    """Native Kannada script input must reach the translation adapter untruncated."""
    kannada_prompt = "ನಮಸ್ಕಾರ, ನೀವು ನನಗೆ ಸಹಾಯ ಮಾಡಬಹುದೇ?"
    trans = recorder.make_translation_adapter("Hello, can you help me?")
    guard = recorder.make_classifier_adapter(label="safe")
    target = recorder.make_target_adapter()

    run_protected(
        kannada_prompt,
        prompt_id="kannada-test",
        translation_adapter=trans,
        classifier_adapter=guard,
        target_adapter=target,
    )

    assert recorder.translation_calls == [kannada_prompt]


def test_kanglish_input_reaches_translation_adapter(recorder: MockCallRecorder):
    """Kanglish / romanized Kannada input must reach the translation adapter accurately."""
    kanglish_prompt = "En guru hegiddiya, nange ond help bekittu"
    trans = recorder.make_translation_adapter("Hey friend how are you, I needed a help")
    guard = recorder.make_classifier_adapter(label="safe")
    target = recorder.make_target_adapter()

    run_protected(
        kanglish_prompt,
        prompt_id="kanglish-test",
        translation_adapter=trans,
        classifier_adapter=guard,
        target_adapter=target,
    )

    assert recorder.translation_calls == [kanglish_prompt]


# ============================================================================
# 17. Prompt ID Is Preserved
# ============================================================================

def test_prompt_id_is_preserved(recorder: MockCallRecorder):
    """The caller-supplied prompt_id must be strictly preserved across all routes."""
    test_id = "test-uuid-98765-abcd"
    target = recorder.make_target_adapter()
    trans = recorder.make_translation_adapter()
    guard = recorder.make_classifier_adapter(label="safe")

    res_base = run_baseline("Prompt 1", prompt_id=test_id, target_adapter=target)
    assert res_base["prompt_id"] == test_id

    res_prot = run_protected(
        "Prompt 2",
        prompt_id=test_id,
        translation_adapter=trans,
        classifier_adapter=guard,
        target_adapter=target,
    )
    assert res_prot["prompt_id"] == test_id


# ============================================================================
# 18. Model Names and Latency Fields Are Present
# ============================================================================

def test_model_names_and_latency_fields_present(recorder: MockCallRecorder):
    """Model names and non-negative latency_ms measurements must be present."""
    trans = recorder.make_translation_adapter(model_name="indic-trans-v2")
    guard = recorder.make_classifier_adapter(label="safe", model_name="llama-guard-custom")
    target = recorder.make_target_adapter(model_name="vaksha-target-llm")

    res = run_protected(
        "ಕನ್ನಡ ಪಠ್ಯ",
        prompt_id="perf-check",
        translation_adapter=trans,
        classifier_adapter=guard,
        target_adapter=target,
    )

    assert res["target_model"] == "vaksha-target-llm"
    assert isinstance(res["latency_ms"], float)
    assert res["latency_ms"] >= 0.0

    assert res["translation"]["model"] == "indic-trans-v2"
    assert isinstance(res["translation"]["latency_ms"], float)
    assert res["translation"]["latency_ms"] >= 0.0

    assert res["guard"]["model"] == "llama-guard-custom"
    assert isinstance(res["guard"]["latency_ms"], float)
    assert res["guard"]["latency_ms"] >= 0.0


# ============================================================================
# 19. Blocked and Review-Required Requests Have Null Target Responses
# ============================================================================

def test_blocked_and_review_required_have_null_target_responses(recorder: MockCallRecorder):
    """Under no circumstance should blocked or review-required requests have non-null target responses."""
    trans = recorder.make_translation_adapter()
    target = recorder.make_target_adapter("Secret target answer")

    # Case A: Blocked
    guard_unsafe = recorder.make_classifier_adapter(label="unsafe")
    res_blocked = run_protected(
        "Prompt",
        translation_adapter=trans,
        classifier_adapter=guard_unsafe,
        target_adapter=target,
    )
    assert res_blocked["target_response"] is None
    assert res_blocked["target_called"] is False

    # Case B: Review required (unknown)
    guard_unknown = recorder.make_classifier_adapter(label="unknown")
    res_review_unk = run_protected(
        "Prompt",
        translation_adapter=trans,
        classifier_adapter=guard_unknown,
        target_adapter=target,
    )
    assert res_review_unk["target_response"] is None
    assert res_review_unk["target_called"] is False

    # Case C: Review required (translation failed)
    def failing_trans(t: str):
        raise RuntimeError("Trans failure")

    res_review_trans = run_protected(
        "Prompt",
        translation_adapter=failing_trans,
        classifier_adapter=guard_unsafe,
        target_adapter=target,
    )
    assert res_review_trans["target_response"] is None
    assert res_review_trans["target_called"] is False

    # Target model must never have been called in any of these cases
    assert len(recorder.target_calls) == 0
