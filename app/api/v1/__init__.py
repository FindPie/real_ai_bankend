from fastapi import APIRouter

from app.api.v1.endpoints import chat, files, models, speech

api_router = APIRouter()

api_router.include_router(chat.router, prefix="/chat", tags=["Chat"])
api_router.include_router(files.router, prefix="/files", tags=["Files"])
api_router.include_router(models.router, prefix="/models", tags=["Models"])
api_router.include_router(speech.router, prefix="/speech", tags=["Speech"])
