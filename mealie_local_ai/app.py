from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from mealie_local_ai.config import Settings
from mealie_local_ai.food_resolver import FoodResolver
from mealie_local_ai.handlers.base import Handler
from mealie_local_ai.handlers.ingredient_parsing import IngredientParsingHandler
from mealie_local_ai.mealie_client import MealieClient
from mealie_local_ai.model_manager import ModelManager
from mealie_local_ai.models import ChatCompletionRequest
from mealie_local_ai.proxy import ProxyHandler
from mealie_local_ai.router import Router

logger = logging.getLogger(__name__)

_HANDLER_REGISTRY: dict[str, Handler] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    mealie_client = MealieClient(
        base_url=settings.MEALIE_URL,
        api_key=settings.MEALIE_API_KEY,
        cache_ttl=settings.MEALIE_CACHE_TTL,
    )

    models_config: dict[str, str | None] = {
        "ingredient_extractor": settings.MODEL_INGREDIENT_EXTRACTOR,
        "general": settings.MODEL_GENERAL,
    }
    model_manager = ModelManager(
        models=models_config,
        strategy=settings.MODEL_LOADING_STRATEGY,
        n_ctx=settings.MODEL_CONTEXT_SIZE,
        n_threads=settings.MODEL_THREADS,
        cache_dir=settings.MODEL_CACHE_DIR,
    )

    if settings.MODEL_LOADING_STRATEGY == "all":
        await model_manager.load_all()

    router = Router(threshold=settings.ROUTER_THRESHOLD)
    proxy = ProxyHandler(
        upstream_url=settings.UPSTREAM_URL,
        upstream_api_key=settings.UPSTREAM_API_KEY,
        timeout=settings.UPSTREAM_TIMEOUT,
    )

    food_resolver = FoodResolver(model_name=settings.MODEL_INGREDIENT_RESOLVER)
    ingredient_handler = IngredientParsingHandler(
        food_resolver=food_resolver,
        model_id=settings.MODEL_INGREDIENT_EXTRACTOR,
    )
    _HANDLER_REGISTRY["parse-recipe-ingredients"] = ingredient_handler

    app.state.settings = settings
    app.state.mealie_client = mealie_client
    app.state.model_manager = model_manager
    app.state.router = router
    app.state.proxy = proxy

    yield

    await model_manager.close()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.delete("/v1/cache")
async def clear_cache(request: Request):
    await request.app.state.mealie_client.clear_cache()
    return {"status": "cleared"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.body()
    chat_request = ChatCompletionRequest.model_validate_json(body)

    system_msg = chat_request.system_message
    matched_name = None
    if system_msg:
        matched_name = request.app.state.router.match(system_msg)

    handler = _HANDLER_REGISTRY.get(matched_name) if matched_name else None

    if handler:
        async with request.app.state.model_manager.get_model(handler.model_key) as model:
            response = await handler.handle(chat_request, model, request.app.state.mealie_client)
        return response.model_dump()

    content_type = request.headers.get("content-type", "application/json")
    return await request.app.state.proxy.forward_chat_completion(body, content_type, chat_request.stream)


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(request: Request):
    body = await request.body()
    content_type = request.headers.get("content-type", "multipart/form-data")
    return await request.app.state.proxy.forward_audio_transcription(body, content_type)
