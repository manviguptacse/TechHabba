
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.model import (
    generate,
    check_ollama,
    ModelServiceError,
)
from core.pipeline import run_baseline, run_protected
from core.gemma_adapters import register_gemma_adapters


app = FastAPI(
    title="BhashaShield API",
    description="Cross-lingual safety gateway for Kannada and English prompts",
    version="1.0.0",
)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10000)
    prompt_id: str | None = None


@app.on_event("startup")
async def startup_event():
    register_gemma_adapters()


@app.get("/health")
async def health():
    status = await check_ollama()

    if not status["ollama_connected"] or not status["model_available"]:
        raise HTTPException(status_code=503, detail=status)

    return {"status": "healthy", **status}


@app.post("/generate")
async def generate_endpoint(request: GenerateRequest):
    try:
        return await generate(request.prompt)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ModelServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/baseline")
async def baseline_endpoint(request: GenerateRequest):
    result = run_baseline(
        prompt=request.prompt,
        prompt_id=request.prompt_id,
    )
    return result


@app.post("/protected")
async def protected_endpoint(request: GenerateRequest):
    result = run_protected(
        prompt=request.prompt,
        prompt_id=request.prompt_id,
    )
    return result
