"""
MCP服务中心配置
"""
from typing import Optional
from pydantic_settings import BaseSettings


class MCPSettings(BaseSettings):
    """MCP服务配置"""

    # 服务基本信息
    service_name: str = "Real AI MCP Server"
    service_version: str = "1.0.0"
    service_description: str = "Real AI MCP服务中心 - 提供统一的MCP服务注册和调用"

    # 服务器配置
    host: str = "localhost"
    port: int = 8100

    # 日志配置
    log_level: str = "INFO"

    # 服务注册配置
    auto_discover_services: bool = True  # 自动发现services目录下的服务

    class Config:
        env_prefix = "MCP_"
        case_sensitive = False


# 全局配置实例
mcp_settings = MCPSettings()
