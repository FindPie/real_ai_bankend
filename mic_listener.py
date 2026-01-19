#!/usr/bin/env python3
"""
麦克风实时监听脚本

从麦克风捕获音频并通过 WebSocket 发送到后端进行实时语音识别。
支持 TTS 语音合成播放。

用法:
    python mic_listener.py [--device DEVICE_INDEX] [--host HOST] [--port PORT]

Docker 音频设备配置:
    需要在 docker-compose.yml 中映射音频设备:
    devices:
      - "/dev/snd/controlC1:/dev/snd/controlC1"
      - "/dev/snd/pcmC1D0c:/dev/snd/pcmC1D0c"
      - "/dev/snd/pcmC1D0p:/dev/snd/pcmC1D0p"
"""

import argparse
import asyncio
import base64
import json
import queue
import signal
import struct
import sys
import threading
from typing import Optional

import pyaudio
import websockets
from loguru import logger

# 音频配置
FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1  # 单声道

# 设备采样率 (麦克风实际采样率)
DEVICE_RATE = 48000

# 目标采样率 (DashScope 支持 8000 和 16000)
TARGET_RATE = 16000

# 重采样比例
RESAMPLE_RATIO = DEVICE_RATE // TARGET_RATE  # 48000 / 16000 = 3

# 每次读取的帧数 (按设备采样率计算，约 200ms 的数据)
# 48000 * 0.2 = 9600 帧，重采样后变成 3200 帧
CHUNK = 9600

# WebSocket 配置
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000
WS_PATH = "/api/v1/speech/recognize/stream"

# TTS 播放配置
TTS_SAMPLE_RATE = 48000  # TTS 输出采样率 (与服务端一致)
TTS_CHANNELS = 1
TTS_FORMAT = pyaudio.paInt16

# 唤醒词配置
DEFAULT_WAKE_WORD = "贾维斯"
DEFAULT_WAKE_WORD_ENABLED = True


class AudioPlayer:
    """音频播放器 - 用于播放 TTS 合成的语音"""

    def __init__(self, audio: pyaudio.PyAudio):
        self.audio = audio
        self.stream: Optional[pyaudio.Stream] = None
        self.audio_queue: queue.Queue = queue.Queue()
        self.running = False
        self._play_thread: Optional[threading.Thread] = None
        self.is_playing = False  # 标记是否正在播放

    def start(self) -> None:
        """启动播放器"""
        if self.stream is not None:
            return

        try:
            self.stream = self.audio.open(
                format=TTS_FORMAT,
                channels=TTS_CHANNELS,
                rate=TTS_SAMPLE_RATE,
                output=True,
                frames_per_buffer=4096,
            )
            self.running = True

            # 启动播放线程
            self._play_thread = threading.Thread(target=self._play_loop, daemon=True)
            self._play_thread.start()

            logger.info(f"音频播放器已启动 (采样率: {TTS_SAMPLE_RATE}Hz)")
        except Exception as e:
            logger.error(f"启动音频播放器失败: {e}")

    def _play_loop(self) -> None:
        """播放循环 (在单独线程中运行)"""
        import time
        while self.running:
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                if audio_data is None:  # 停止信号
                    break
                if self.stream:
                    self.is_playing = True  # 标记正在播放
                    self.stream.write(audio_data)
            except queue.Empty:
                if self.is_playing:
                    # 播放完成后延迟200ms再恢复监听，避免残余回声
                    time.sleep(0.2)
                    self.is_playing = False  # 队列空，播放完成
                continue
            except Exception as e:
                logger.error(f"播放音频时出错: {e}")
                self.is_playing = False

    def play(self, audio_data: bytes) -> None:
        """将音频数据加入播放队列"""
        if self.running:
            self.audio_queue.put(audio_data)

    def stop(self) -> None:
        """停止播放器"""
        self.running = False
        self.is_playing = False
        self.audio_queue.put(None)  # 发送停止信号

        if self._play_thread:
            self._play_thread.join(timeout=1.0)
            self._play_thread = None

        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        # 清空队列
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        logger.info("音频播放器已停止")


def resample_audio(audio_data: bytes, ratio: int) -> bytes:
    """
    简单重采样：每隔 ratio 个采样点取一个

    Args:
        audio_data: 原始音频数据 (16-bit PCM)
        ratio: 重采样比例 (例如 3 表示 48000 -> 16000)

    Returns:
        重采样后的音频数据
    """
    # 将 bytes 转换为 16-bit 整数列表
    samples = struct.unpack(f"<{len(audio_data) // 2}h", audio_data)

    # 每隔 ratio 个采样点取一个
    resampled = samples[::ratio]

    # 转换回 bytes
    return struct.pack(f"<{len(resampled)}h", *resampled)


class MicrophoneListener:
    """麦克风监听器"""

    def __init__(
        self,
        device_index: Optional[int] = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        enable_tts: bool = True,
        wake_word_enabled: bool = DEFAULT_WAKE_WORD_ENABLED,
        wake_word: str = DEFAULT_WAKE_WORD,
    ):
        self.device_index = device_index
        self.ws_url = f"ws://{host}:{port}{WS_PATH}"
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.running = False
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.enable_tts = enable_tts
        self.wake_word_enabled = wake_word_enabled
        self.wake_word = wake_word
        self.audio_player: Optional[AudioPlayer] = None

    def list_devices(self) -> None:
        """列出所有可用的音频输入设备"""
        logger.info("可用的音频输入设备:")
        info = self.audio.get_host_api_info_by_index(0)
        num_devices = info.get("deviceCount", 0)

        for i in range(num_devices):
            device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
            if device_info.get("maxInputChannels", 0) > 0:
                name = device_info.get("name", "Unknown")
                default_rate = int(device_info.get("defaultSampleRate", 0))
                logger.info(f"  [{i}] {name} (默认采样率: {default_rate}Hz)")

    def get_default_input_device(self) -> Optional[int]:
        """获取默认输入设备索引"""
        try:
            default_info = self.audio.get_default_input_device_info()
            return default_info.get("index")
        except IOError:
            logger.warning("无法获取默认输入设备")
            return None

    async def connect_websocket(self) -> bool:
        """连接 WebSocket"""
        try:
            logger.info(f"正在连接 WebSocket: {self.ws_url}")
            self.websocket = await websockets.connect(self.ws_url)
            logger.info("WebSocket 连接成功")

            # 发送开始命令 (使用目标采样率)
            start_msg = {
                "action": "start",
                "format": "pcm",
                "sample_rate": TARGET_RATE,
                "enable_tts": self.enable_tts,
                "wake_word_enabled": self.wake_word_enabled,
                "wake_word": self.wake_word,
            }
            await self.websocket.send(json.dumps(start_msg))
            if self.wake_word_enabled:
                logger.info(f"唤醒词已启用: '{self.wake_word}'")

            # 等待确认
            response = await self.websocket.recv()
            resp_data = json.loads(response)
            if resp_data.get("status") == "started":
                logger.info("语音识别已启动")
                return True
            else:
                logger.error(f"启动失败: {resp_data}")
                return False

        except Exception as e:
            logger.error(f"WebSocket 连接失败: {e}")
            return False

    async def receive_results(self) -> None:
        """接收识别结果和大模型回复的协程"""
        llm_response = ""
        try:
            while self.running and self.websocket:
                try:
                    response = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=0.1
                    )
                    data = json.loads(response)

                    msg_type = data.get("type", "")

                    # TTS 准备信号
                    if msg_type == "tts_prepare":
                        logger.debug("[TTS] 准备播放音频")
                        # 启动音频播放器
                        if self.enable_tts and self.audio_player:
                            self.audio_player.start()

                    # 语音识别结果
                    elif msg_type == "recognition":
                        text = data.get("text", "")
                        is_final = data.get("is_final", False)

                        if is_final:
                            logger.success(f"[语音识别] {text}")
                        else:
                            logger.info(f"[实时识别] {text}")

                    # 大模型回复
                    elif msg_type == "llm":
                        if data.get("status") == "thinking":
                            logger.info("[AI] 正在思考...")
                            llm_response = ""
                            # 启动音频播放器 (准备接收 TTS 音频)
                            if self.enable_tts and self.audio_player:
                                self.audio_player.start()
                        elif data.get("error"):
                            logger.error(f"[AI 错误] {data['error']}")
                        elif data.get("done"):
                            # 回复完成
                            print()  # 换行
                            logger.success(f"[AI 回复完成]")
                        else:
                            # 流式输出
                            chunk = data.get("content", "")
                            llm_response += chunk
                            print(chunk, end="", flush=True)

                    # TTS 音频数据
                    elif msg_type == "tts_audio":
                        if data.get("done"):
                            logger.debug("[TTS] 音频播放完成")
                        else:
                            # 解码并播放音频
                            audio_data = data.get("data", "")
                            if audio_data and self.audio_player:
                                audio_bytes = base64.b64decode(audio_data)
                                self.audio_player.play(audio_bytes)

                    # 唤醒词检测
                    elif msg_type == "wake_word":
                        if data.get("status") == "activated":
                            wake_word = data.get("wake_word", "")
                            logger.success(f"[唤醒词] 检测到 '{wake_word}'，已激活!")

                    elif "error" in data:
                        logger.error(f"错误: {data['error']}")

                    elif "status" in data:
                        status = data["status"]
                        if status == "started":
                            logger.info(f"识别已启动 (LLM: {data.get('enable_llm', True)})")
                        elif status == "stopped":
                            logger.debug("识别已停止")
                        else:
                            logger.debug(f"状态: {status}")

                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket 连接已关闭")
                    break

        except Exception as e:
            logger.error(f"接收结果时出错: {e}")

    async def send_audio(self) -> None:
        """发送音频数据的协程"""
        try:
            while self.running and self.stream and self.websocket:
                try:
                    # 检查是否正在播放TTS音频
                    if self.audio_player and self.audio_player.is_playing:
                        # 播放期间，仍然读取音频但不发送，避免缓冲区溢出
                        self.stream.read(CHUNK, exception_on_overflow=False)
                        await asyncio.sleep(0.01)
                        continue

                    # 读取音频数据 (48kHz)
                    data = self.stream.read(CHUNK, exception_on_overflow=False)

                    # 重采样到 16kHz
                    resampled_data = resample_audio(data, RESAMPLE_RATIO)

                    # 发送重采样后的数据
                    await self.websocket.send(resampled_data)
                    await asyncio.sleep(0.01)  # 小延迟避免过载

                except IOError as e:
                    logger.warning(f"音频读取错误: {e}")
                    await asyncio.sleep(0.1)
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket 连接已关闭")
                    break

        except Exception as e:
            logger.error(f"发送音频时出错: {e}")

    async def start(self) -> None:
        """启动麦克风监听"""
        # 确定使用的设备
        device = self.device_index
        if device is None:
            device = self.get_default_input_device()
            if device is None:
                # 尝试查找第一个可用的输入设备
                self.list_devices()
                logger.error("请使用 --device 参数指定设备索引")
                return

        logger.info(f"使用音频设备索引: {device}")

        # 打开音频流 (使用设备原生采样率)
        try:
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=DEVICE_RATE,
                input=True,
                input_device_index=device,
                frames_per_buffer=CHUNK,
            )
            logger.info(f"音频流已打开 (设备采样率: {DEVICE_RATE}Hz -> 目标: {TARGET_RATE}Hz)")

        except IOError as e:
            logger.error(f"无法打开音频设备: {e}")
            self.list_devices()
            return

        # 初始化音频播放器 (用于 TTS)
        if self.enable_tts:
            self.audio_player = AudioPlayer(self.audio)
            logger.info("TTS 音频播放器已初始化")

        # 连接 WebSocket
        if not await self.connect_websocket():
            self.stream.close()
            return

        self.running = True
        logger.info("开始监听麦克风... 输入 'h' 查看帮助, Ctrl+C 停止")

        # 并发运行发送、接收和键盘处理
        try:
            await asyncio.gather(
                self.send_audio(),
                self.receive_results(),
                self.handle_keyboard(),
            )
        except asyncio.CancelledError:
            logger.info("任务已取消")

    async def handle_keyboard(self) -> None:
        """处理键盘输入"""
        import sys
        import select

        while self.running:
            try:
                # 非阻塞检查标准输入
                if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                    line = sys.stdin.readline().strip().lower()
                    if line == "q" or line == "quit":
                        self.running = False
                        break
                    elif line == "h" or line == "help":
                        print("\n命令帮助:")
                        print("  q/quit  - 退出程序")
                        print("  h/help  - 显示帮助\n")
                else:
                    await asyncio.sleep(0.1)
            except Exception:
                await asyncio.sleep(0.1)

    async def stop(self) -> None:
        """停止监听"""
        self.running = False
        logger.info("正在停止...")

        # 发送停止命令
        if self.websocket:
            try:
                await self.websocket.send(json.dumps({"action": "stop"}))
                # 等待最终结果
                try:
                    response = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=2.0
                    )
                    data = json.loads(response)
                    if "text" in data:
                        logger.success(f"[最终结果] {data['text']}")
                except asyncio.TimeoutError:
                    pass
                await self.websocket.close()
            except Exception as e:
                logger.debug(f"关闭 WebSocket 时出错: {e}")

        # 停止音频播放器
        if self.audio_player:
            self.audio_player.stop()
            self.audio_player = None

        # 关闭音频流
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        self.audio.terminate()
        logger.info("已停止")

    def cleanup(self) -> None:
        """清理资源 (同步版本，用于信号处理)"""
        self.running = False
        if self.audio_player:
            self.audio_player.stop()
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
        self.audio.terminate()


async def main():
    parser = argparse.ArgumentParser(description="麦克风实时语音识别")
    parser.add_argument(
        "--device", "-d",
        type=int,
        default=None,
        help="音频输入设备索引"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"WebSocket 服务器地址 (默认: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=DEFAULT_PORT,
        help=f"WebSocket 服务器端口 (默认: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--list-devices", "-l",
        action="store_true",
        help="列出可用的音频设备"
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="禁用 TTS 语音合成播放"
    )
    parser.add_argument(
        "--no-wake-word",
        action="store_true",
        help="禁用唤醒词检测"
    )
    parser.add_argument(
        "--wake-word", "-w",
        type=str,
        default=DEFAULT_WAKE_WORD,
        help=f"设置唤醒词 (默认: {DEFAULT_WAKE_WORD})"
    )

    args = parser.parse_args()

    listener = MicrophoneListener(
        device_index=args.device,
        host=args.host,
        port=args.port,
        enable_tts=not args.no_tts,
        wake_word_enabled=not args.no_wake_word,
        wake_word=args.wake_word,
    )

    if args.list_devices:
        listener.list_devices()
        listener.audio.terminate()
        return

    # 设置信号处理
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("收到停止信号")
        asyncio.create_task(listener.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await listener.start()
    except KeyboardInterrupt:
        pass
    finally:
        await listener.stop()


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="DEBUG",
    )

    asyncio.run(main())
