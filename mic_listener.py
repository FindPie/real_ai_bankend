#!/usr/bin/env python3
"""
麦克风实时监听脚本

从麦克风捕获音频并通过 WebSocket 发送到后端进行实时语音识别。

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
import json
import signal
import struct
import sys
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
    ):
        self.device_index = device_index
        self.ws_url = f"ws://{host}:{port}{WS_PATH}"
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.running = False
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None

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
            }
            await self.websocket.send(json.dumps(start_msg))

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
        """接收识别结果的协程"""
        try:
            while self.running and self.websocket:
                try:
                    response = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=0.1
                    )
                    data = json.loads(response)

                    if "text" in data:
                        text = data["text"]
                        is_final = data.get("is_final", False)
                        confidence = data.get("confidence", 0)

                        if is_final:
                            logger.success(f"[最终结果] {text} (置信度: {confidence:.2f})")
                        else:
                            logger.info(f"[实时识别] {text}")

                    elif "error" in data:
                        logger.error(f"识别错误: {data['error']}")

                    elif "status" in data:
                        logger.debug(f"状态: {data['status']}")

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

        # 连接 WebSocket
        if not await self.connect_websocket():
            self.stream.close()
            return

        self.running = True
        logger.info("开始监听麦克风... 按 Ctrl+C 停止")

        # 并发运行发送和接收
        try:
            await asyncio.gather(
                self.send_audio(),
                self.receive_results(),
            )
        except asyncio.CancelledError:
            logger.info("任务已取消")

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

        # 关闭音频流
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        self.audio.terminate()
        logger.info("已停止")

    def cleanup(self) -> None:
        """清理资源 (同步版本，用于信号处理)"""
        self.running = False
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

    args = parser.parse_args()

    listener = MicrophoneListener(
        device_index=args.device,
        host=args.host,
        port=args.port,
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
