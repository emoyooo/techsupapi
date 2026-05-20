import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.models import GenerateRequest, GenerateResponse, HealthResponse
from app.inference import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_PATH    = os.getenv("ADAPTER_PATH", "/kaggle/working/fine_tuned_model")

model_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager
    logger.info("Loading model...")
    model_manager = ModelManager(BASE_MODEL_NAME, ADAPTER_PATH)
    try:
        model_manager.load()
    except Exception as e:
        logger.error("Failed to load model: %s", e)
    yield
    model_manager = None

app = FastAPI(
    title="Support Ticket Classifier API",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health", response_model=HealthResponse)
async def health():
    loaded = model_manager is not None and model_manager.is_loaded
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_name=BASE_MODEL_NAME,
        adapter_path=ADAPTER_PATH,
    )

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if model_manager is None or not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    try:
        result = await model_manager.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return GenerateResponse(
        generated_text=result,
        model=BASE_MODEL_NAME,
        prompt_length=len(request.prompt),
        generated_length=len(result),
    )
