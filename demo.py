"""Demo: demo.py
Role: Member 3 - VAKSHA
Purpose: Demonstration script for baseline and protected routes
         using strictly controlled mock adapters.
"""

from typing import Any, Dict
from core.pipeline import run_baseline, run_protected


# ============================================================================
# Controlled Mock Adapters (MOCKED DEMONSTRATIONS ONLY)
# ============================================================================

def mock_translation(text: str) -> Dict[str, Any]:
    """Mock translator mapping test prompts to English translations."""
    translations = {
        "ನಮಸ್ಕಾರ, ಇವತ್ತಿನ ಹವಾಮಾನ ಹೇಗಿದೆ?": "Hello, how is the weather today?",
        "ಬೆಂಗಳೂರಿನ ಪ್ರಸಿದ್ಧ ತಾಣಗಳು ಯಾವುವು?": "What are the famous spots in Bengaluru?",
        "ಕೃತಕ ಪರೀಕ್ಷಾ ಹಾನಿಕಾರಕ ಪ್ರಾಂಪ್ಟ್ [MOCK]": "Synthetic test unsafe prompt [MOCK]",
        "ದೋಷ ಪರೀಕ್ಷಾ ಪ್ರಾಂಪ್ಟ್": "Error test prompt",
    }
    return {
        "translated_text": translations.get(text, f"Translated: {text}"),
        "model": "mock-indic-translator-v1",
    }


def mock_classifier(text: str) -> Dict[str, Any]:
    """Mock safety classifier identifying safe vs unsafe test prompts."""
    if "unsafe" in text.lower():
        return {
            "label": "unsafe",
            "category": "synthetic_policy_violation",
            "model": "mock-safety-guard-v1",
        }
    return {
        "label": "safe",
        "category": None,
        "model": "mock-safety-guard-v1",
    }


def mock_target(prompt: str) -> Dict[str, Any]:
    """Mock target LLM returning a canned response to the original prompt."""
    return {
        "response": f"Mock LLM generated response for prompt: '{prompt}'",
        "model": "mock-target-llm-v1",
    }


def print_demo_result(title: str, result: Dict[str, Any]) -> None:
    """Helper to format and display pipeline output for demonstrations."""
    print("=" * 72)
    print(f"MOCKED DEMONSTRATION: {title}")
    print("=" * 72)
    print(f"  Input Prompt             : {result.get('original_prompt')}")

    trans = result.get("translation")
    trans_text = trans.get("translated_text") if trans else "N/A (Not invoked)"
    print(f"  Translation Result       : {trans_text}")

    guard = result.get("guard")
    guard_label = guard.get("label") if guard else "N/A (Not invoked)"
    print(f"  Guard Label              : {guard_label}")

    print(f"  Decision                 : {result.get('decision')}")
    print(f"  Target Model Called      : {result.get('target_called')}")
    print(f"  Final Status             : {result.get('status')}")
    if result.get("errors"):
        print(f"  Errors / Notes           : {result.get('errors')}")
    print()


def main():
    print("\n" + "#" * 72)
    print("  VAKSHA PIPELINE DEMONSTRATION (MOCKED DEMONSTRATIONS ONLY)")
    print("  Note: No live models, real API calls, or external services used.")
    print("#" * 72 + "\n")

    # 1. Baseline Request
    base_prompt = "ನಮಸ್ಕಾರ, ಇವತ್ತಿನ ಹವಾಮಾನ ಹೇಗಿದೆ?"
    res_base = run_baseline(
        prompt=base_prompt,
        prompt_id="demo-baseline-001",
        target_adapter=mock_target,
    )
    print_demo_result("1. Baseline Request (Direct to Target)", res_base)

    # 2. Protected Safe Request
    safe_prompt = "ಬೆಂಗಳೂರಿನ ಪ್ರಸಿದ್ಧ ತಾಣಗಳು ಯಾವುವು?"
    res_safe = run_protected(
        prompt=safe_prompt,
        prompt_id="demo-protected-safe-002",
        translation_adapter=mock_translation,
        classifier_adapter=mock_classifier,
        target_adapter=mock_target,
    )
    print_demo_result("2. Protected Safe Request (ALLOW -> Target Called)", res_safe)

    # 3. Protected Unsafe Request
    unsafe_prompt = "ಕೃತಕ ಪರೀಕ್ಷಾ ಹಾನಿಕಾರಕ ಪ್ರಾಂಪ್ಟ್ [MOCK]"
    res_unsafe = run_protected(
        prompt=unsafe_prompt,
        prompt_id="demo-protected-unsafe-003",
        translation_adapter=mock_translation,
        classifier_adapter=mock_classifier,
        target_adapter=mock_target,
    )
    print_demo_result("3. Protected Unsafe Request (BLOCK -> Target NOT Called)", res_unsafe)

    # 4. Translation Failure
    def failing_translation(text: str):
        raise TimeoutError("Mock translation gateway timed out after 5000ms")

    fail_prompt = "ದೋಷ ಪರೀಕ್ಷಾ ಪ್ರಾಂಪ್ಟ್"
    res_fail = run_protected(
        prompt=fail_prompt,
        prompt_id="demo-protected-fail-004",
        translation_adapter=failing_translation,
        classifier_adapter=mock_classifier,
        target_adapter=mock_target,
    )
    print_demo_result("4. Translation Failure (REVIEW_REQUIRED -> Target NOT Called)", res_fail)


if __name__ == "__main__":
    main()
