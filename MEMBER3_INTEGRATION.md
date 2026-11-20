# VAKSHA - Member 3 Integration Guide

This document defines the interface contracts, module architecture, and integration points for Member 3's core modules.

---

## 1. Purpose of Assigned Files

| File | Status | Purpose |
| :--- | :--- | :--- |
| `core/translate.py` | **Implemented** (Mock-ready) | Translates Kannada/Kanglish prompts to English via an injectable adapter. Measures latency and handles errors. |
| `core/guard.py` | **Implemented** (Mock-ready) | Evaluates translated text for safety. Returns normalized labels (`safe`, `unsafe`, `unknown`). Never converts errors to safe. |
| `core/pipeline.py` | **Implemented** (Mock-ready) | Coordinates `baseline` and `protected` execution flows. Enforces safety decisions (`ALLOW`, `BLOCK`, `REVIEW_REQUIRED`). |
| `tests/test_pipeline.py` | **Implemented** (19/19 Passing) | Fully offline, deterministic test suite using mock adapters. Zero API keys or network dependencies required. |

> **Note on Live Integrations**: Live model clients and API keys are **pending** integration from Member 1. All core modules are built with injectable adapters to run seamlessly with mocks or live functions.

---

## 2. Function Signatures

### Translation (`core/translate.py`)
```python
def translate_to_english(text: str, adapter: Optional[Callable] = None) -> dict: ...
def set_translation_adapter(adapter: Callable) -> None: ...
```

### Safety Classifier (`core/guard.py`)
```python
def classify_prompt(text: str, adapter: Optional[Callable] = None) -> dict: ...
def set_classifier_adapter(adapter: Callable) -> None: ...
```

### Pipelines (`core/pipeline.py`)
```python
def run_baseline(
    prompt: str,
    prompt_id: Optional[Any] = None,
    target_adapter: Optional[Callable] = None,
) -> dict: ...

def run_protected(
    prompt: str,
    prompt_id: Optional[Any] = None,
    translation_adapter: Optional[Callable] = None,
    classifier_adapter: Optional[Callable] = None,
    target_adapter: Optional[Callable] = None,
) -> dict: ...

def set_target_adapter(adapter: Callable) -> None: ...
```

---

## 3. Translation Adapter Contract (Member 1)
- **Input**: `text: str` (Kannada script, romanized Kannada, or Kanglish).
- **Return Value**:
  - `str` (the English translation), **OR**
  - `dict` with keys `{"translated_text": str, "model": Optional[str]}`.
- **Failures**: Should raise `TimeoutError` or `Exception` on connection/API errors.

---

## 4. Safety Classifier Adapter Contract (Member 1)
- **Input**: `text: str` (English translated prompt).
- **Return Value**:
  - Plain string (`"safe"`, `"unsafe"`, `"unknown"`), **OR**
  - JSON string / dictionary containing:
    ```python
    {
        "label": "safe" | "unsafe" | "unknown",
        "category": Optional[str],  # e.g., "hate_speech", "jailbreak", or None
        "model": Optional[str]
    }
    ```
- **Failures**: On timeout or exception, the module safely marks the result as `unknown` with `decision="REVIEW_REQUIRED"`.

---

## 5. Target-Model Adapter Contract (Member 1)
- **Input**: `prompt: str` (**CRITICAL**: Always receives the original Kannada/Kanglish prompt, never translated text).
- **Return Value**:
  - `str` (the raw response), **OR**
  - `dict` containing `{"response": str, "model": Optional[str]}`.
- **Failures**: Raises `Exception` or `TimeoutError` if model inference fails.

---

## 6. Pipeline Result Dictionary Schema

Every execution of `run_baseline()` and `run_protected()` returns a standardized dictionary:

```python
{
    "prompt_id": Any,                 # Preserved identifier from input
    "original_prompt": str,           # The exact prompt passed in
    "route": "baseline" | "protected",
    "status": "success" | "blocked" | "review_required" | "failed",
    "decision": "ALLOW" | "BLOCK" | "REVIEW_REQUIRED",
    "translation": dict | None,       # Translation output and latency (None in baseline)
    "guard": dict | None,             # Guard output, label, category (None in baseline)
    "target_model": str | None,       # Identifier of the target model called
    "target_response": str | None,    # Target reply (Always None if blocked or review_required)
    "target_called": bool,            # True only if target model was actually invoked
    "latency_ms": float,              # Total round-trip execution latency in ms
    "errors": list[str]               # Diagnostic messages if errors occurred
}
```

---

## 7. How to Run Pytest

Run the deterministic test suite from the project root:

```bash
python -m pytest tests/test_pipeline.py -v
```

All 19 test cases run offline using mock adapters without requiring credentials.

---

## 8. Requirements from Member 1 (Model Provider)
1. **Provide 3 callable functions** matching the contracts in Sections 3, 4, and 5:
   - Translation function.
   - Guard / Safety classifier function.
   - Target AI model function.
2. **Register adapters**: Call `set_translation_adapter()`, `set_classifier_adapter()`, and `set_target_adapter()` on startup, or pass them into pipeline calls.
3. **Environment & Secrets**: Member 1 manages API credentials; Member 3 files do not store keys.

---

## 9. Integration Requirements from Member 2 (Dataset Provider)
1. **Prompt Batching**: Member 2 can iterate through dataset prompts and call:
   - `run_baseline(prompt=row["prompt"], prompt_id=row["id"])`
   - `run_protected(prompt=row["prompt"], prompt_id=row["id"])`
2. **Language Support**: Prompts can be native Kannada script, romanized Kannada, or Kanglish.

---

## 10. Expected Output Fields for Member 4 (Evaluator / Analyst)
Member 4 can consume the pipeline output dictionary to compute project metrics:
- **Safety & Filtering**: Inspect `decision` (`ALLOW`, `BLOCK`, `REVIEW_REQUIRED`) and `guard["category"]`.
- **Target Accuracy**: Read `target_response` (guaranteed `None` for blocked/review items).
- **Attack Prevention**: Check `target_called` (guaranteed `False` for unsafe prompts).
- **Latency Impact**: Compare `latency_ms` between baseline and protected routes.
- **Error Rates**: Analyze `status` and `errors`.
