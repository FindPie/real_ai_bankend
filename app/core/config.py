from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # 应用基础配置
    app_name: str = "Real AI Backend"
    app_version: str = "0.1.0"
    debug: bool = False

    # API 配置
    api_v1_prefix: str = "/api/v1"

    # CORS 配置
    cors_origins: List[str] = ["*"]

    # 数据库配置
    database_url: str = "sqlite+aiosqlite:///./real_ai.db"

    # Redis 配置 (可选)
    redis_url: str = "redis://localhost:6379/0"

    # JWT 配置
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # OpenRouter API 配置
    openrouter_api_key: str = ""
    openrouter_api_url: str = "https://openrouter.ai/api/v1/chat/completions"

    # 阿里云语音服务配置
    aliyun_access_key_id: str = ""
    aliyun_access_key_secret: str = ""
    aliyun_speech_app_key: str = ""


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()


settings = get_settings()
