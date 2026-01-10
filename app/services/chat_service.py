from typing import AsyncGenerator, List, Optional

import httpx
from loguru import logger

from app.core.config import settings
from app.schemas.chat import ChatResponse, Message


class ChatService:
    """聊天服务"""

    def __init__(self):
        self.api_url = settings.openrouter_api_url
        self.api_key = settings.openrouter_api_key

    async def send_message(
        self,
        messages: List[Message],
        model: str,
        web_search: bool = False,
    ) -> ChatResponse:
        """发送消息到 OpenRouter API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://real-ai.app",
            "X-Title": "Real AI Backend",
        }

        request_body = {
            "model": model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
        }

        if web_search:
            request_body["plugins"] = [{"id": "web"}]

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                self.api_url,
                headers=headers,
                json=request_body,
            )

            if response.status_code != 200:
                error_data = response.json() if response.content else {}
                error_msg = error_data.get("error", {}).get("message", f"API 请求失败: {response.status_code}")
                raise Exception(error_msg)

            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage")

            return ChatResponse(
                content=content,
                model=model,
                usage=usage,
            )

    async def send_message_stream(
        self,
        messages: List[Message],
        model: str,
        web_search: bool = False,
    ) -> AsyncGenerator[str, None]:
        """流式发送消息到 OpenRouter API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://real-ai.app",
            "X-Title": "Real AI Backend",
        }

        request_body = {
            "model": model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "stream": True,
        }

        if web_search:
            request_body["plugins"] = [{"id": "web"}]

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                self.api_url,
                headers=headers,
                json=request_body,
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise Exception(f"API 请求失败: {response.status_code}")

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            import json
                            parsed = json.loads(data)
                            content = parsed.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                yield content
                        except Exception as e:
                            logger.debug(f"解析流式数据失败: {e}")
                            continue


chat_service = ChatService()
