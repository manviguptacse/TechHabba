# VAKSHA Integration Notes (Member 3 -> Member 1)

This workspace covers Member 3's assigned modules: `core/translate.py`, `core/guard.py`, `core/pipeline.py`, and `tests/test_pipeline.py`.

## Interfaces Required from Member 1:

1. **AI / Model Client Interface:**
   - How `client` or model calls should be instantiated and passed into `translate_text()` and `check_guardrails()`.
   - The expected method signature (e.g., `client.generate(...)`, `client.predict(...)`, or an API wrapper).

2. **Configuration & Credentials:**
   - Centralized environment variable conventions (e.g., `.env` keys) or config objects for model endpoints, API keys, and language defaults.

3. **Response & Error Schema:**
   - Standard output format (dictionary or dataclass) for guardrail check results (e.g., `{"is_safe": bool, "flags": list}`).
   - Standard exception types or error handling conventions across VAKSHA.
