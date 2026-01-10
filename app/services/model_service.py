from typing import List

from app.schemas.model import ModelInfo, ModelListResponse


# 可用模型列表
AVAILABLE_MODELS: List[ModelInfo] = [
    ModelInfo(id="google/gemini-3-pro-preview", name="Gemini 3 Pro（贵）", provider="Google", type="vision"),
    ModelInfo(id="google/gemini-3-pro-image-preview", name="Gemini 3 Pro Image（贵）", provider="Google", type="image-gen"),
    ModelInfo(id="google/gemini-3-flash-preview", name="Gemini 3 Flash", provider="Google", type="vision"),
    ModelInfo(id="google/gemini-2.5-flash-lite", name="Gemini 2.5 Flash Lite", provider="Google", type="vision"),
    ModelInfo(id="openai/gpt-5.2", name="GPT-5.2（贵）", provider="OpenAI", type="chat"),
    ModelInfo(id="openai/gpt-5.2-chat", name="GPT-5.2 Chat（贵）", provider="OpenAI", type="vision"),
    ModelInfo(id="openai/gpt-5-mini", name="GPT-5 Mini", provider="OpenAI", type="vision"),
    ModelInfo(id="openai/gpt-5-image-mini", name="GPT-5 Image Mini", provider="OpenAI", type="image-gen"),
    ModelInfo(id="anthropic/claude-opus-4.5", name="Claude Opus 4.5（贵）", provider="Anthropic", type="vision"),
    ModelInfo(id="anthropic/claude-haiku-4.5", name="Claude Haiku 4.5", provider="Anthropic", type="vision"),
    ModelInfo(id="qwen/qwen3-235b-a22b-thinking-2507", name="Qwen3 235B Thinking", provider="Alibaba", type="chat"),
    ModelInfo(id="qwen/qwen3-vl-235b-a22b-instruct", name="Qwen3 VL 235B [视觉]", provider="Alibaba", type="vision"),
    ModelInfo(id="deepseek/deepseek-v3.2", name="DeepSeek V3.2", provider="DeepSeek", type="chat"),
]


class ModelService:
    """模型服务"""

    def get_all_models(self) -> ModelListResponse:
        """获取所有可用模型"""
        return ModelListResponse(
            models=AVAILABLE_MODELS,
            total=len(AVAILABLE_MODELS),
        )

    def get_model_by_id(self, model_id: str) -> ModelInfo | None:
        """根据 ID 获取模型信息"""
        for model in AVAILABLE_MODELS:
            if model.id == model_id:
                return model
        return None

    def get_models_by_provider(self, provider: str) -> List[ModelInfo]:
        """根据提供商获取模型列表"""
        return [m for m in AVAILABLE_MODELS if m.provider.lower() == provider.lower()]

    def get_models_by_type(self, model_type: str) -> List[ModelInfo]:
        """根据类型获取模型列表"""
        return [m for m in AVAILABLE_MODELS if m.type == model_type]


model_service = ModelService()
