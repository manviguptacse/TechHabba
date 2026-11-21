
import time

import httpx

from config import settings


OLLAMA_BASE_URL = (
    settings.model_api_base_url or "http://localhost:11434"
).rstrip("/")

TARGET_MODEL = settings.target_model or "gemma3:1b"

TIMEOUT = settings.model_timeout_seconds


class ModelServiceError(Exception):
    """Raised when the local model service fails."""


async def generate(prompt: str) -> dict:
    """
    Generate a response using the configured Ollama model.

    Returns:
        text, model name, and measured latency.
    """

    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    payload = {
        "model": TARGET_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    start_time = time.perf_counter()

    try:
        async with httpx.AsyncClient(
            timeout=TIMEOUT
        ) as client:

            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
            )

            response.raise_for_status()
            result = response.json()

    except httpx.TimeoutException as exc:
        raise ModelServiceError(
            "Model request timed out."
        ) from exc

    except httpx.HTTPStatusError as exc:
        raise ModelServiceError(
            f"Ollama returned HTTP {exc.response.status_code}."
        ) from exc

    except httpx.RequestError as exc:
        raise ModelServiceError(
            "Cannot connect to Ollama. Check whether it is running."
        ) from exc

    except ValueError as exc:
        raise ModelServiceError(
            "Ollama returned an invalid response."
        ) from exc

    latency_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2,
    )

    text = result.get("response", "").strip()

    if not text:
        raise ModelServiceError(
            "The model returned an empty response."
        )

    return {
        "text": text,
        "model": TARGET_MODEL,
        "latency_ms": latency_ms,
    }


async def check_ollama() -> dict:
    """Check Ollama availability and target-model installation."""

    try:
        async with httpx.AsyncClient(
            timeout=5.0
        ) as client:

            response = await client.get(
                f"{OLLAMA_BASE_URL}/api/tags"
            )

            response.raise_for_status()
            data = response.json()

        installed_models = [
            model.get("name")
            for model in data.get("models", [])
        ]

        model_available = (
            TARGET_MODEL in installed_models
        )

        return {
            "ollama_connected": True,
            "model": TARGET_MODEL,
            "model_available": model_available,
        }

    except (httpx.HTTPError, ValueError):
        return {
            "ollama_connected": False,
            "model": TARGET_MODEL,
            "model_available": False,
        }
