"""
MCP服务中心 - 主服务器
"""
import importlib
import pkgutil
from pathlib import Path
from typing import Any, Dict

from fastmcp import FastMCP
from loguru import logger

from mcp_center.config import mcp_settings
from mcp_center.registry import registry


# 创建 FastMCP 应用
mcp = FastMCP(
    name=mcp_settings.service_name,
)


def auto_discover_services():
    """自动发现并加载 services 目录下的所有服务"""
    services_dir = Path(__file__).parent / "services"

    if not services_dir.exists():
        logger.warning(f"服务目录不存在: {services_dir}")
        return

    logger.info("开始自动发现MCP服务...")

    # 遍历 services 目录下的所有 Python 模块
    for importer, modname, ispkg in pkgutil.iter_modules([str(services_dir)]):
        if modname.startswith("_"):
            continue

        try:
            # 动态导入模块
            module_path = f"mcp_center.services.{modname}"
            module = importlib.import_module(module_path)

            # 调用模块的 initialize 函数（如果存在）
            if hasattr(module, "initialize"):
                module.initialize()
                logger.info(f"✓ 加载服务模块: {modname}")
            else:
                logger.warning(f"模块 {modname} 没有 initialize() 函数")

        except Exception as e:
            logger.error(f"✗ 加载服务模块 {modname} 失败: {e}")


def register_mcp_tools():
    """将注册中心的所有工具注册到 FastMCP"""
    tools = registry.get_all_tools()

    for tool in tools:
        service_name = tool["service"]
        tool_name = tool["name"]
        description = tool["description"]
        parameters = tool["parameters"]

        # 获取处理函数
        handler = registry.get_tool_handler(service_name, tool_name)
        if not handler:
            logger.warning(f"工具 {service_name}.{tool_name} 没有处理函数")
            continue

        # 注册到 FastMCP
        full_tool_name = f"{service_name}_{tool_name}"

        @mcp.tool(name=full_tool_name, description=description)
        async def tool_wrapper(arguments: Dict[str, Any], handler=handler):
            """工具包装器"""
            try:
                # 同步调用处理函数
                result = handler(**arguments)
                return result
            except Exception as e:
                logger.error(f"工具执行失败: {e}")
                return {"error": str(e)}

        logger.info(f"  ✓ 注册MCP工具: {full_tool_name}")


def register_mcp_prompts():
    """将注册中心的所有提示词注册到 FastMCP"""
    prompts = registry.get_all_prompts()

    for prompt in prompts:
        service_name = prompt["service"]
        prompt_name = prompt["name"]
        description = prompt["description"]
        arguments = prompt["arguments"]

        # 获取处理函数
        handler = registry.get_prompt_handler(service_name, prompt_name)
        if not handler:
            logger.warning(f"提示词 {service_name}.{prompt_name} 没有处理函数")
            continue

        # 注册到 FastMCP
        full_prompt_name = f"{service_name}_{prompt_name}"

        @mcp.prompt(name=full_prompt_name, description=description)
        async def prompt_wrapper(arguments: Dict[str, Any], handler=handler):
            """提示词包装器"""
            try:
                # 同步调用处理函数
                result = handler(**arguments)
                return result
            except Exception as e:
                logger.error(f"提示词执行失败: {e}")
                return f"Error: {e}"

        logger.info(f"  ✓ 注册MCP提示词: {full_prompt_name}")


def register_mcp_resources():
    """将注册中心的所有资源注册到 FastMCP"""
    resources = registry.get_all_resources()

    for resource in resources:
        service_name = resource["service"]
        resource_uri = resource["uri"]
        description = resource["description"]
        mime_type = resource["mime_type"]

        # 获取处理函数
        handler = registry.get_resource_handler(resource_uri)
        if not handler:
            logger.warning(f"资源 {resource_uri} 没有处理函数")
            continue

        @mcp.resource(uri=resource_uri, name=description, mime_type=mime_type)
        async def resource_wrapper(handler=handler):
            """资源包装器"""
            try:
                # 同步调用处理函数
                result = handler()
                return result
            except Exception as e:
                logger.error(f"资源获取失败: {e}")
                return {"error": str(e)}

        logger.info(f"  ✓ 注册MCP资源: {resource_uri}")


def initialize_server():
    """初始化 MCP 服务器"""
    logger.info("=" * 60)
    logger.info("正在初始化 Real AI MCP 服务中心...")
    logger.info("=" * 60)

    # 1. 自动发现并加载服务
    if mcp_settings.auto_discover_services:
        auto_discover_services()

    # 2. 注册工具到 FastMCP
    logger.info("\n注册工具到 MCP...")
    register_mcp_tools()

    # 3. 注册提示词到 FastMCP
    logger.info("\n注册提示词到 MCP...")
    register_mcp_prompts()

    # 4. 注册资源到 FastMCP
    # TODO: FastMCP 的资源注册需要 URI 模板参数，暂时跳过
    # logger.info("\n注册资源到 MCP...")
    # register_mcp_resources()

    # 5. 显示统计信息
    services = registry.list_services()
    logger.info("\n" + "=" * 60)
    logger.info(f"✓ MCP 服务中心初始化完成!")
    logger.info(f"  - 已加载服务: {len(services)}")
    logger.info(f"  - 已注册工具: {len(registry.get_all_tools())}")
    logger.info(f"  - 已注册提示词: {len(registry.get_all_prompts())}")
    logger.info(f"  - 已注册资源: {len(registry.get_all_resources())}")
    logger.info("=" * 60)


# 初始化服务器
initialize_server()
