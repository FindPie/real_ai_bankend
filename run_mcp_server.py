#!/usr/bin/env python3
"""
MCP服务中心启动脚本
"""
import sys
from loguru import logger

from mcp_center.config import mcp_settings
from mcp_center.server import mcp


def setup_logging():
    """配置日志"""
    logger.remove()  # 移除默认处理器
    logger.add(
        sys.stdout,
        level=mcp_settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    )


def main():
    """主函数"""
    setup_logging()

    logger.info(f"启动 {mcp_settings.service_name} v{mcp_settings.service_version}")
    logger.info(f"监听地址: {mcp_settings.host}:{mcp_settings.port}")

    # 运行 FastMCP 服务器
    # FastMCP 使用标准输入/输出进行通信（stdio transport）
    # 不需要指定 host 和 port，因为 MCP 协议通过 stdio 工作
    mcp.run()


if __name__ == "__main__":
    main()
