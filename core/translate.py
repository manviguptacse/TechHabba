"""Module: core/translate.py
Role: Member 3 - VAKSHA
Purpose: Translates Kannada, romanized Kannada, and Kanglish text to English
         using an injectable translation-model adapter provided by Member 1.
"""

from typing import Any, Callable, Dict, Optional, Union
import time


class TranslationError(Exception):
    """Raised when translation fails due to validation, adapter errors, or timeouts."""
    pass


# Type alias for translation adapter: accepts input text, returns translated str or dict
TranslationAdapter = Callable[[str], Union[str, Dict[str, Any]]]

# Registered default adapter (to be connected by Member 1 during project integration)
_default_adapter: Optional[TranslationAdapter] = None


def set_translation_adapter(adapter: TranslationAdapter) -> None:
    """Connect Member 1's actual translation model function as the default adapter.

    :param adapter: A callable accepting text (str) and returning str or dict.
    """
    global _default_adapter
    _default_adapter = adapter


def get_translation_adapter() -> Optional[TranslationAdapter]:
    """Retrieve the currently registered default translation adapter."""
    return _default_adapter


def translate_to_english(
    text: str,
    adapter: Optional[TranslationAdapter] = None,
) -> Dict[str, Any]:
    """Translates Kannada, romanized Kannada, or Kanglish input text to English.

    Uses an injectable translation adapter. If no adapter is provided and no default
    is registered, signals that live integration with Member 1 is pending.

    :param text: Input prompt in Kannada script, romanized Kannada, or Kanglish.
    :param adapter: Optional translation-model adapter callable to override the default.
    :return: A dictionary with keys:
             - 'translated_text': The translated English string.
             - 'model': The model identifier or adapter name.
             - 'latency_ms': Execution latency in milliseconds.
    :raises TranslationError: On empty input, adapter absence, timeout, empty response, or API failure.
    """
    # 1. Validate input text
    if not isinstance(text, str):
        raise TranslationError("Input text must be a valid string.")

    cleaned_text = text.strip()
    if not cleaned_text:
        raise TranslationError("Input text cannot be empty or whitespace only.")

    # 2. Resolve translation adapter
    active_adapter = adapter if adapter is not None else _default_adapter

    if active_adapter is None:
        raise TranslationError(
            "Live integration pending: Translation model adapter is not configured. "
            "Member 1's model function must be connected via the adapter."
        )

    # 3. Execute adapter and measure latency
    start_time = time.perf_counter()

    try:
        raw_response = active_adapter(cleaned_text)
    except TimeoutError as exc:
        raise TranslationError(f"Translation request timed out: {exc}") from exc
    except Exception as exc:
        raise TranslationError(f"Translation API error: {exc}") from exc

    end_time = time.perf_counter()
    latency_ms = round((end_time - start_time) * 1000, 2)

    # 4. Parse adapter response (supports raw string or dictionary response)
    translated_text = ""
    model_name = getattr(active_adapter, "__name__", "external_translation_model")

    if isinstance(raw_response, dict):
        if "error" in raw_response and raw_response["error"]:
            raise TranslationError(f"Translation model returned error: {raw_response['error']}")
        translated_text = raw_response.get("translated_text") or raw_response.get("text") or ""
        model_name = raw_response.get("model", model_name)
    elif isinstance(raw_response, str):
        translated_text = raw_response.strip()
    else:
        raise TranslationError(f"Unexpected response format from translation adapter: {type(raw_response)}")

    # 5. Validate translation output
    if not translated_text:
        raise TranslationError("Translation model returned an empty response.")

    return {
        "translated_text": translated_text,
        "model": str(model_name),
        "latency_ms": latency_ms,
    }
