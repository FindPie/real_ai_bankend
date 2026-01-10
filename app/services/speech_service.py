import base64
import io
from typing import Optional

from loguru import logger

from app.core.config import settings
from app.schemas.speech import SpeechToTextResponse, TextToSpeechResponse


class SpeechService:
    """语音服务 - 支持阿里云语音识别"""

    def __init__(self):
        self.aliyun_access_key_id = settings.aliyun_access_key_id
        self.aliyun_access_key_secret = settings.aliyun_access_key_secret
        self.aliyun_app_key = settings.aliyun_speech_app_key

    async def speech_to_text(
        self,
        audio_data: str,
        audio_format: str = "wav",
        sample_rate: int = 16000,
        language: str = "zh-CN",
    ) -> SpeechToTextResponse:
        """
        语音转文字

        Args:
            audio_data: Base64 编码的音频数据
            audio_format: 音频格式 (pcm, wav, mp3, m4a, webm, ogg)
            sample_rate: 采样率
            language: 语言代码

        Returns:
            SpeechToTextResponse: 识别结果
        """
        try:
            # 解码 Base64 音频数据
            audio_bytes = base64.b64decode(audio_data)
            logger.info(f"接收到音频数据: {len(audio_bytes)} bytes, 格式: {audio_format}")

            # TODO: 集成阿里云语音识别 API
            # 目前返回占位响应，等待阿里云 API 配置
            text = await self._call_aliyun_asr(
                audio_bytes, audio_format, sample_rate, language
            )

            return SpeechToTextResponse(
                text=text,
                duration=len(audio_bytes) / (sample_rate * 2),  # 估算时长
                confidence=0.95,
            )

        except Exception as e:
            logger.error(f"语音识别失败: {e}")
            raise Exception(f"语音识别失败: {str(e)}")

    async def _call_aliyun_asr(
        self,
        audio_bytes: bytes,
        audio_format: str,
        sample_rate: int,
        language: str,
    ) -> str:
        """
        调用阿里云语音识别 API

        阿里云一句话识别 API 文档:
        https://help.aliyun.com/document_detail/92131.html
        """
        # 检查配置
        if not self.aliyun_access_key_id or not self.aliyun_access_key_secret:
            logger.warning("阿里云 AccessKey 未配置，使用模拟响应")
            return "[阿里云语音识别未配置，请在 .env 中设置 ALIYUN_ACCESS_KEY_ID 和 ALIYUN_ACCESS_KEY_SECRET]"

        try:
            # 使用阿里云 NLS SDK
            import nls

            # 创建识别请求
            sr = nls.NlsSpeechRecognizer(
                url="wss://nls-gateway.cn-shanghai.aliyuncs.com/ws/v1",
                akid=self.aliyun_access_key_id,
                aksecret=self.aliyun_access_key_secret,
                appkey=self.aliyun_app_key,
            )

            result_text = ""

            def on_result(message, *args):
                nonlocal result_text
                result_text = message

            sr.on_result_changed = on_result
            sr.on_sentence_end = on_result

            # 开始识别
            sr.start(
                aformat=audio_format,
                sample_rate=sample_rate,
                enable_punctuation_prediction=True,
                enable_inverse_text_normalization=True,
            )

            # 发送音频数据
            sr.send_audio(audio_bytes)
            sr.stop()

            return result_text or "无法识别"

        except ImportError:
            logger.warning("阿里云 NLS SDK 未安装，使用 HTTP API 方式")
            return await self._call_aliyun_asr_http(
                audio_bytes, audio_format, sample_rate
            )
        except Exception as e:
            logger.error(f"阿里云语音识别调用失败: {e}")
            raise

    async def _call_aliyun_asr_http(
        self,
        audio_bytes: bytes,
        audio_format: str,
        sample_rate: int,
    ) -> str:
        """
        使用 HTTP 方式调用阿里云语音识别 (一句话识别 RESTful API)
        """
        import hashlib
        import hmac
        import time
        import uuid
        from urllib.parse import quote

        import httpx

        # 阿里云一句话识别 API 地址
        url = "https://nls-gateway.cn-shanghai.aliyuncs.com/stream/v1/asr"

        # 请求参数
        params = {
            "appkey": self.aliyun_app_key,
            "format": audio_format,
            "sample_rate": sample_rate,
            "enable_punctuation_prediction": "true",
            "enable_inverse_text_normalization": "true",
        }

        # 构建签名 (简化版，实际需要完整的阿里云签名)
        headers = {
            "Content-Type": f"audio/{audio_format}; samplerate={sample_rate}",
            "X-NLS-Token": await self._get_aliyun_token(),
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                params=params,
                headers=headers,
                content=audio_bytes,
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("result", "无法识别")
            else:
                logger.error(f"阿里云 ASR 请求失败: {response.status_code} {response.text}")
                raise Exception(f"语音识别请求失败: {response.status_code}")

    async def _get_aliyun_token(self) -> str:
        """获取阿里云 NLS Token"""
        import httpx

        url = "https://nls-meta.cn-shanghai.aliyuncs.com/"

        # 这里需要使用阿里云签名机制获取 Token
        # 简化实现，实际需要完整的签名
        async with httpx.AsyncClient() as client:
            # TODO: 实现完整的阿里云签名获取 Token
            pass

        return ""

    async def text_to_speech(
        self,
        text: str,
        voice: str = "xiaoyun",
        speed: int = 0,
        volume: int = 50,
        audio_format: str = "mp3",
    ) -> TextToSpeechResponse:
        """
        文字转语音

        Args:
            text: 要转换的文字
            voice: 发音人
            speed: 语速
            volume: 音量
            audio_format: 输出音频格式

        Returns:
            TextToSpeechResponse: 合成的音频
        """
        # TODO: 集成阿里云语音合成 API
        logger.info(f"文字转语音: {text[:50]}...")

        # 返回占位响应
        return TextToSpeechResponse(
            audio_data="",
            audio_format=audio_format,
            duration=len(text) * 0.3,  # 估算时长
        )


speech_service = SpeechService()
