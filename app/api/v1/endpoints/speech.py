import base64
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from loguru import logger

from app.schemas.speech import (
    SpeechToTextRequest,
    SpeechToTextResponse,
    TextToSpeechRequest,
    TextToSpeechResponse,
)
from app.schemas.chat import Message
from app.services.chat_service import chat_service
from app.services.speech_service import speech_service

router = APIRouter()

# 默认大模型配置 (OpenRouter 模型 ID)
DEFAULT_LLM_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_WEB_SEARCH = True  # 默认开启联网搜索
DEFAULT_ENABLE_TTS = True  # 默认开启语音合成

# TTS 配置
DEFAULT_TTS_MODEL = "cosyvoice-v3-flash"
DEFAULT_TTS_VOICE = "longxiaochun_v3"

# 唤醒词配置
WAKE_WORD = "贾维斯"
WAKE_WORD_ENABLED = True  # 默认启用唤醒词

# 支持的音频格式
SUPPORTED_AUDIO_FORMATS = {"pcm", "wav", "mp3", "m4a", "webm", "ogg", "flac", "amr"}
MAX_AUDIO_SIZE = 10 * 1024 * 1024  # 10MB


@router.post(
    "/recognize",
    response_model=SpeechToTextResponse,
    summary="语音识别 (Base64)",
    description="将 Base64 编码的音频数据转换为文字",
)
async def recognize_speech(request: SpeechToTextRequest) -> SpeechToTextResponse:
    """
    语音识别 - Base64 方式

    接收 Base64 编码的音频数据，返回识别的文字

    - **audio_data**: Base64 编码的音频数据
    - **audio_format**: 音频格式 (pcm, wav, mp3, m4a, webm, ogg)
    - **sample_rate**: 采样率，默认 16000
    - **language**: 语言代码，默认 zh-CN
    """
    try:
        # 验证 Base64 数据
        try:
            audio_bytes = base64.b64decode(request.audio_data)
        except Exception:
            raise HTTPException(status_code=400, detail="无效的 Base64 音频数据")

        # 检查音频大小
        if len(audio_bytes) > MAX_AUDIO_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"音频文件过大，最大支持 {MAX_AUDIO_SIZE // 1024 // 1024}MB",
            )

        result = await speech_service.speech_to_text(
            audio_data=request.audio_data,
            audio_format=request.audio_format,
            sample_rate=request.sample_rate,
            language=request.language,
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"语音识别失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/recognize/upload",
    response_model=SpeechToTextResponse,
    summary="语音识别 (文件上传)",
    description="上传音频文件进行语音识别",
)
async def recognize_speech_upload(
    file: UploadFile = File(..., description="音频文件"),
    sample_rate: int = Form(default=16000, description="采样率"),
    language: str = Form(default="zh-CN", description="语言代码"),
) -> SpeechToTextResponse:
    """
    语音识别 - 文件上传方式

    直接上传音频文件进行识别

    支持的格式: pcm, wav, mp3, m4a, webm, ogg, flac, amr
    文件大小限制: 10MB
    """
    # 检查文件名
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    # 获取文件扩展名
    extension = file.filename.split(".")[-1].lower() if "." in file.filename else ""

    if extension not in SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的音频格式: {extension}。支持: {', '.join(SUPPORTED_AUDIO_FORMATS)}",
        )

    # 读取文件内容
    content = await file.read()

    if len(content) > MAX_AUDIO_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"音频文件过大，最大支持 {MAX_AUDIO_SIZE // 1024 // 1024}MB",
        )

    # 转换为 Base64
    audio_base64 = base64.b64encode(content).decode("utf-8")

    try:
        result = await speech_service.speech_to_text(
            audio_data=audio_base64,
            audio_format=extension,
            sample_rate=sample_rate,
            language=language,
        )
        return result

    except Exception as e:
        logger.error(f"语音识别失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/recognize/stream")
async def recognize_speech_stream(websocket: WebSocket):
    """
    实时语音识别 + 大模型对话 - WebSocket 流式接口

    用于接收麦克风实时音频流，实时返回识别结果，并在识别完成后调用大模型

    协议:
    1. 客户端发送 JSON 配置: {"action": "start", "format": "pcm", "sample_rate": 16000, "enable_llm": true}
    2. 客户端发送二进制音频数据 (持续)
    3. 服务端返回识别结果: {"type": "recognition", "text": "识别的文字", "is_final": false}
    4. 客户端发送: {"action": "stop"} 结束识别
    5. 如果启用 LLM，服务端返回: {"type": "llm", "content": "大模型回复", "done": false}
    6. LLM 回复完成: {"type": "llm", "content": "", "done": true}
    """
    await websocket.accept()
    logger.info("WebSocket 连接已建立")

    import asyncio
    import json
    import queue

    recognizer = None
    audio_format = "pcm"
    sample_rate = 16000
    enable_llm = True  # 默认启用大模型
    llm_model = DEFAULT_LLM_MODEL
    web_search = DEFAULT_WEB_SEARCH  # 是否启用联网搜索
    enable_tts = DEFAULT_ENABLE_TTS  # 是否启用语音合成
    tts_voice = DEFAULT_TTS_VOICE  # TTS 声音
    wake_word_enabled = WAKE_WORD_ENABLED  # 是否启用唤醒词
    wake_word = WAKE_WORD  # 唤醒词
    is_activated = False  # 是否已被唤醒
    accumulated_text = ""  # 累积的识别文本
    conversation_history: list = []  # 对话历史

    # 使用队列在线程间传递识别结果
    result_queue: queue.Queue = queue.Queue()

    async def process_results():
        """从队列中读取结果并发送给客户端"""
        while True:
            try:
                # 非阻塞方式检查队列
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: result_queue.get(timeout=0.1)
                )
                if result is None:  # 停止信号
                    break
                await websocket.send_json(result)
            except queue.Empty:
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"发送结果失败: {e}")
                break

    # 启动结果处理任务
    result_task = asyncio.create_task(process_results())

    try:
        while True:
            data = await websocket.receive()

            # 处理文本消息 (控制命令)
            if "text" in data:
                message = json.loads(data["text"])
                action = message.get("action")

                if action == "start":
                    audio_format = message.get("format", "pcm")
                    sample_rate = message.get("sample_rate", 16000)
                    enable_llm = message.get("enable_llm", True)
                    llm_model = message.get("model", DEFAULT_LLM_MODEL)
                    web_search = message.get("web_search", DEFAULT_WEB_SEARCH)
                    enable_tts = message.get("enable_tts", DEFAULT_ENABLE_TTS)
                    tts_voice = message.get("tts_voice", DEFAULT_TTS_VOICE)
                    wake_word_enabled = message.get("wake_word_enabled", WAKE_WORD_ENABLED)
                    wake_word = message.get("wake_word", WAKE_WORD)
                    is_activated = not wake_word_enabled  # 如果禁用唤醒词，则直接激活
                    accumulated_text = ""

                    # 创建实时识别器
                    try:
                        def on_result(text: str, is_final: bool):
                            # 在回调线程中将结果放入队列
                            result_queue.put({
                                "type": "recognition",
                                "text": text,
                                "is_final": is_final,
                            })

                        recognizer = speech_service.create_realtime_recognizer(
                            on_result=on_result,
                            sample_rate=sample_rate,
                            audio_format=audio_format,
                        )
                        recognizer.start()
                        logger.info(f"开始实时识别: format={audio_format}, sample_rate={sample_rate}, enable_llm={enable_llm}")
                        await websocket.send_json({"status": "started", "enable_llm": enable_llm})

                    except Exception as e:
                        logger.error(f"启动识别器失败: {e}")
                        await websocket.send_json({"error": str(e)})

                elif action == "stop":
                    final_text = ""
                    if recognizer:
                        final_text = recognizer.stop() or ""
                        if final_text:
                            await websocket.send_json({
                                "type": "recognition",
                                "text": final_text,
                                "is_final": True,
                            })
                        recognizer = None

                    logger.info(f"实时识别结束, 最终文本: {final_text}")
                    await websocket.send_json({"status": "stopped"})

                    # 累积识别文本
                    accumulated_text += final_text

                    # 唤醒词检测逻辑
                    user_query = ""
                    if wake_word_enabled:
                        if not is_activated:
                            # 检查是否包含唤醒词
                            if wake_word in accumulated_text:
                                is_activated = True
                                # 提取唤醒词后的内容
                                wake_idx = accumulated_text.find(wake_word)
                                user_query = accumulated_text[wake_idx + len(wake_word):].strip()
                                logger.info(f"检测到唤醒词 '{wake_word}'，已激活")
                                await websocket.send_json({
                                    "type": "wake_word",
                                    "status": "activated",
                                    "wake_word": wake_word,
                                })
                                accumulated_text = ""  # 清空累积文本
                            else:
                                logger.debug(f"等待唤醒词 '{wake_word}'...")
                        else:
                            # 已激活状态，直接使用识别结果
                            user_query = final_text.strip()
                            is_activated = False  # 处理完后重置激活状态
                            accumulated_text = ""
                    else:
                        # 未启用唤醒词，直接使用识别结果
                        user_query = final_text.strip()

                    # 如果启用了大模型且有用户查询，调用大模型
                    if enable_llm and user_query:
                        tts_synthesizer = None
                        tts_audio_task = None

                        try:
                            logger.info(f"调用大模型: {llm_model}, 联网搜索: {web_search}, TTS: {enable_tts}, 查询: {user_query[:50]}...")
                            await websocket.send_json({
                                "type": "llm",
                                "status": "thinking",
                                "query": user_query,
                            })

                            # 如果启用 TTS，创建语音合成器
                            if enable_tts:
                                try:
                                    tts_synthesizer = speech_service.create_realtime_tts(
                                        model=DEFAULT_TTS_MODEL,
                                        voice=tts_voice,
                                    )
                                    tts_synthesizer.start()

                                    # 启动音频发送任务
                                    async def send_tts_audio():
                                        """从 TTS 队列读取音频并发送给客户端"""
                                        audio_queue = tts_synthesizer.get_audio_queue()
                                        while True:
                                            try:
                                                audio_data = await asyncio.get_event_loop().run_in_executor(
                                                    None, lambda: audio_queue.get(timeout=0.1)
                                                )
                                                if audio_data is None:  # 结束信号
                                                    break
                                                # 发送音频数据给客户端
                                                await websocket.send_json({
                                                    "type": "tts_audio",
                                                    "data": base64.b64encode(audio_data).decode("utf-8"),
                                                    "done": False,
                                                })
                                            except queue.Empty:
                                                await asyncio.sleep(0.01)
                                            except Exception as e:
                                                logger.error(f"发送 TTS 音频失败: {e}")
                                                break

                                        # 发送音频完成信号
                                        await websocket.send_json({
                                            "type": "tts_audio",
                                            "done": True,
                                        })

                                    tts_audio_task = asyncio.create_task(send_tts_audio())
                                    logger.info(f"TTS 合成器已启动 (声音: {tts_voice})")

                                except Exception as e:
                                    logger.error(f"创建 TTS 合成器失败: {e}")
                                    tts_synthesizer = None

                            # 添加用户消息到对话历史
                            conversation_history.append(Message(role="user", content=user_query))

                            # 流式调用大模型
                            full_response = ""
                            async for chunk in chat_service.send_message_stream(
                                messages=conversation_history,
                                model=llm_model,
                                web_search=web_search,
                            ):
                                full_response += chunk
                                await websocket.send_json({
                                    "type": "llm",
                                    "content": chunk,
                                    "done": False,
                                })

                                # 同时发送给 TTS 合成器
                                if tts_synthesizer:
                                    tts_synthesizer.send_text(chunk)

                            # 完成 TTS 合成
                            if tts_synthesizer:
                                tts_synthesizer.complete()
                                # 等待音频发送完成
                                if tts_audio_task:
                                    await tts_audio_task

                            # 添加助手回复到对话历史
                            conversation_history.append(Message(role="assistant", content=full_response))

                            await websocket.send_json({
                                "type": "llm",
                                "content": "",
                                "done": True,
                                "full_response": full_response,
                            })
                            logger.info(f"大模型回复完成: {full_response[:100]}...")

                        except Exception as e:
                            logger.error(f"大模型调用失败: {e}")
                            await websocket.send_json({
                                "type": "llm",
                                "error": str(e),
                            })
                            # 清理 TTS 资源
                            if tts_audio_task and not tts_audio_task.done():
                                tts_audio_task.cancel()

                elif action == "clear_history":
                    # 清空对话历史
                    conversation_history.clear()
                    await websocket.send_json({"status": "history_cleared"})

            # 处理二进制数据 (音频流)
            elif "bytes" in data:
                if recognizer:
                    recognizer.send_audio(data["bytes"])

    except WebSocketDisconnect:
        logger.info("WebSocket 连接已断开")
        if recognizer:
            recognizer.stop()
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")
        if recognizer:
            recognizer.stop()
        await websocket.close(code=1000)
    finally:
        # 停止结果处理任务
        result_queue.put(None)
        result_task.cancel()
        try:
            await result_task
        except asyncio.CancelledError:
            pass


@router.post(
    "/synthesize",
    response_model=TextToSpeechResponse,
    summary="语音合成",
    description="将文字转换为语音",
)
async def synthesize_speech(request: TextToSpeechRequest) -> TextToSpeechResponse:
    """
    语音合成 - 文字转语音

    - **text**: 要转换的文字 (最长 5000 字)
    - **voice**: 发音人 (默认 xiaoyun)
    - **speed**: 语速 (-500 到 500)
    - **volume**: 音量 (0-100)
    - **audio_format**: 输出格式 (pcm, wav, mp3)
    """
    try:
        result = await speech_service.text_to_speech(
            text=request.text,
            voice=request.voice,
            speed=request.speed,
            volume=request.volume,
            audio_format=request.audio_format,
        )
        return result

    except Exception as e:
        logger.error(f"语音合成失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
