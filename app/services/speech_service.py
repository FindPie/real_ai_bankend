import asyncio
import base64
import queue
import threading
from typing import AsyncGenerator, Callable, Optional

import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
from dashscope.audio.tts_v2 import SpeechSynthesizer, ResultCallback, AudioFormat
from loguru import logger

from app.core.config import settings
from app.schemas.speech import SpeechToTextResponse, TextToSpeechResponse


# TTS 配置
TTS_MODEL = "cosyvoice-v3-flash"
TTS_VOICE = "longxiaochun_v3"
TTS_SAMPLE_RATE = 48000


class RealtimeTTSCallback(ResultCallback):
    """实时语音合成回调 - 将音频数据放入队列"""

    def __init__(self, audio_queue: queue.Queue):
        self.audio_queue = audio_queue
        self.is_open = False
        self.error_message: Optional[str] = None

    def on_open(self):
        logger.info("TTS WebSocket 连接已建立")
        self.is_open = True

    def on_complete(self):
        logger.info("TTS 语音合成完成")
        self.audio_queue.put(None)  # 结束信号

    def on_error(self, message: str):
        self.error_message = message
        logger.error(f"TTS 语音合成错误: {message}")
        self.audio_queue.put(None)

    def on_close(self):
        logger.info("TTS WebSocket 连接已关闭")
        self.is_open = False

    def on_event(self, message):
        logger.debug(f"TTS 事件: {message}")

    def on_data(self, data: bytes) -> None:
        """接收到音频数据"""
        self.audio_queue.put(data)


class RealtimeTTSSynthesizer:
    """实时语音合成器 - 流式接收 LLM 文本并合成语音"""

    def __init__(
        self,
        model: str = TTS_MODEL,
        voice: str = TTS_VOICE,
    ):
        self.model = model
        self.voice = voice
        self.audio_queue: queue.Queue = queue.Queue()
        self.callback: Optional[RealtimeTTSCallback] = None
        self.synthesizer: Optional[SpeechSynthesizer] = None

        # 配置 DashScope
        if not settings.dashscope_api_key:
            raise ValueError("DASHSCOPE_API_KEY 未配置")

        dashscope.api_key = settings.dashscope_api_key

    def start(self) -> None:
        """启动语音合成器"""
        self.callback = RealtimeTTSCallback(self.audio_queue)

        self.synthesizer = SpeechSynthesizer(
            model=self.model,
            voice=self.voice,
            format=AudioFormat.PCM_48000HZ_MONO_16BIT,
            callback=self.callback,
        )
        logger.info(f"TTS 合成器已启动 (模型: {self.model}, 声音: {self.voice})")

    def send_text(self, text: str) -> None:
        """发送文本进行合成 (流式)"""
        if self.synthesizer and text:
            self.synthesizer.streaming_call(text)

    def complete(self) -> None:
        """完成文本发送，等待合成结束"""
        if self.synthesizer:
            self.synthesizer.streaming_complete()
            logger.info("TTS 合成完成")

    def get_audio_queue(self) -> queue.Queue:
        """获取音频数据队列"""
        return self.audio_queue

    def get_request_id(self) -> Optional[str]:
        """获取请求 ID"""
        if self.synthesizer:
            return self.synthesizer.get_last_request_id()
        return None


class RealtimeRecognitionCallback(RecognitionCallback):
    """实时语音识别回调"""

    def __init__(self, on_result: Callable[[str, bool], None]):
        self.on_result = on_result
        self.result_queue: queue.Queue = queue.Queue()
        self.is_open = False
        self.error_message: Optional[str] = None

    def on_open(self) -> None:
        logger.info("DashScope 语音识别连接已建立")
        self.is_open = True

    def on_close(self) -> None:
        logger.info("DashScope 语音识别连接已关闭")
        self.is_open = False
        self.result_queue.put(None)  # 结束信号

    def on_complete(self) -> None:
        logger.info("DashScope 语音识别完成")

    def on_error(self, message) -> None:
        self.error_message = str(message.message)
        logger.error(f"DashScope 语音识别错误: {message.message}")
        self.result_queue.put(None)

    def on_event(self, result: RecognitionResult) -> None:
        sentence = result.get_sentence()
        if "text" in sentence:
            text = sentence["text"]
            is_final = RecognitionResult.is_sentence_end(sentence)
            logger.debug(f"识别结果: {text}, is_final: {is_final}")
            self.result_queue.put({"text": text, "is_final": is_final})
            if self.on_result:
                self.on_result(text, is_final)


class RealtimeRecognizer:
    """实时语音识别器"""

    def __init__(
        self,
        on_result: Optional[Callable[[str, bool], None]] = None,
        sample_rate: int = 16000,
        audio_format: str = "pcm",
    ):
        self.sample_rate = sample_rate
        self.audio_format = audio_format
        self.recognition: Optional[Recognition] = None
        self.callback: Optional[RealtimeRecognitionCallback] = None
        self._on_result = on_result

        # 配置 DashScope
        if not settings.dashscope_api_key:
            raise ValueError("DASHSCOPE_API_KEY 未配置")

        dashscope.api_key = settings.dashscope_api_key
        dashscope.base_websocket_api_url = settings.dashscope_websocket_url

    def start(self) -> None:
        """启动语音识别"""
        self.callback = RealtimeRecognitionCallback(on_result=self._on_result)

        self.recognition = Recognition(
            model="paraformer-realtime-v2",
            format=self.audio_format,
            sample_rate=self.sample_rate,
            semantic_punctuation_enabled=False,
            callback=self.callback,
            # 设置句子间最大静音时长(毫秒)，默认800ms，增加到1500ms让用户有更多时间说话
            max_sentence_silence=1500,
        )

        self.recognition.start()
        logger.info(
            f"实时语音识别已启动 (采样率: {self.sample_rate}, 格式: {self.audio_format}, 最大静音: 1500ms)"
        )

    def send_audio(self, audio_data: bytes) -> None:
        """发送音频数据"""
        if self.recognition:
            self.recognition.send_audio_frame(audio_data)

    def stop(self) -> Optional[str]:
        """停止语音识别并返回最终结果"""
        if self.recognition:
            self.recognition.stop()
            logger.info("实时语音识别已停止")

            # 收集所有结果
            final_text = ""
            while True:
                try:
                    result = self.callback.result_queue.get(timeout=1.0)
                    if result is None:
                        break
                    if result.get("is_final"):
                        final_text = result.get("text", "")
                except queue.Empty:
                    break

            return final_text
        return None

    def get_results(self) -> AsyncGenerator[dict, None]:
        """异步获取识别结果"""

        async def _get_results():
            while True:
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: self.callback.result_queue.get(timeout=0.1)
                    )
                    if result is None:
                        break
                    yield result
                except queue.Empty:
                    await asyncio.sleep(0.01)

        return _get_results()


class SpeechService:
    """语音服务 - 支持阿里云 DashScope 语音识别"""

    def __init__(self):
        self.dashscope_api_key = settings.dashscope_api_key
        self.dashscope_websocket_url = settings.dashscope_websocket_url

        # 配置 DashScope (如果有 API Key)
        if self.dashscope_api_key:
            dashscope.api_key = self.dashscope_api_key
            dashscope.base_websocket_api_url = self.dashscope_websocket_url

    def create_realtime_recognizer(
        self,
        on_result: Optional[Callable[[str, bool], None]] = None,
        sample_rate: int = 16000,
        audio_format: str = "pcm",
    ) -> RealtimeRecognizer:
        """创建实时语音识别器"""
        return RealtimeRecognizer(
            on_result=on_result,
            sample_rate=sample_rate,
            audio_format=audio_format,
        )

    def create_realtime_tts(
        self,
        model: str = TTS_MODEL,
        voice: str = TTS_VOICE,
    ) -> RealtimeTTSSynthesizer:
        """创建实时语音合成器"""
        return RealtimeTTSSynthesizer(
            model=model,
            voice=voice,
        )

    async def speech_to_text(
        self,
        audio_data: str,
        audio_format: str = "wav",
        sample_rate: int = 16000,
        language: str = "zh-CN",
    ) -> SpeechToTextResponse:
        """
        语音转文字 (一次性识别)

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

            if not self.dashscope_api_key:
                logger.warning("DASHSCOPE_API_KEY 未配置，返回模拟响应")
                return SpeechToTextResponse(
                    text="[DashScope API Key 未配置，请在 .env 中设置 DASHSCOPE_API_KEY]",
                    duration=len(audio_bytes) / (sample_rate * 2),
                    confidence=0.0,
                )

            # 使用同步方式进行一次性识别
            result_text = await self._recognize_audio(
                audio_bytes, audio_format, sample_rate
            )

            return SpeechToTextResponse(
                text=result_text,
                duration=len(audio_bytes) / (sample_rate * 2),
                confidence=0.95,
            )

        except Exception as e:
            logger.error(f"语音识别失败: {e}")
            raise Exception(f"语音识别失败: {str(e)}")

    async def _recognize_audio(
        self,
        audio_bytes: bytes,
        audio_format: str,
        sample_rate: int,
    ) -> str:
        """使用 DashScope 进行语音识别"""
        result_text = ""
        recognition_done = threading.Event()
        error_message = None

        class OneTimeCallback(RecognitionCallback):
            def on_open(self) -> None:
                logger.debug("一次性识别连接已建立")

            def on_close(self) -> None:
                logger.debug("一次性识别连接已关闭")
                recognition_done.set()

            def on_complete(self) -> None:
                logger.debug("一次性识别完成")
                recognition_done.set()

            def on_error(self, message) -> None:
                nonlocal error_message
                error_message = str(message.message)
                logger.error(f"一次性识别错误: {message.message}")
                recognition_done.set()

            def on_event(self, result: RecognitionResult) -> None:
                nonlocal result_text
                sentence = result.get_sentence()
                if "text" in sentence:
                    result_text = sentence["text"]

        callback = OneTimeCallback()

        recognition = Recognition(
            model="paraformer-realtime-v2",
            format=audio_format,
            sample_rate=sample_rate,
            semantic_punctuation_enabled=True,
            callback=callback,
        )

        # 在线程中运行识别
        def run_recognition():
            recognition.start()
            # 分块发送音频数据
            chunk_size = 3200
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i : i + chunk_size]
                recognition.send_audio_frame(chunk)
            recognition.stop()

        thread = threading.Thread(target=run_recognition)
        thread.start()

        # 等待识别完成
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: recognition_done.wait(timeout=30)
        )
        thread.join(timeout=5)

        if error_message:
            raise Exception(error_message)

        return result_text or "无法识别"

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
            duration=len(text) * 0.3,
        )


speech_service = SpeechService()
