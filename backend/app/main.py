from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes.health import router as health_router
from app.routes.inference import router as inference_router
from app.services.inference_service import InferenceService


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        debug=settings.debug,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = settings
    app.state.inference_service = InferenceService(settings)

    app.include_router(health_router)
    app.include_router(inference_router)

    return app


app = create_app()
