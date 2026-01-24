"""
示例MCP服务 - 演示如何创建MCP服务

这是一个模板服务，展示了如何:
1. 注册服务
2. 添加工具(Tools)
3. 添加提示词(Prompts)
4. 添加资源(Resources)
"""
from loguru import logger
from mcp_center.registry import registry


def initialize():
    """初始化示例服务"""

    # 1. 注册服务
    service_name = "example"
    registry.register_service(
        name=service_name,
        description="示例MCP服务 - 展示基本功能",
        version="1.0.0",
    )

    # 2. 注册工具
    registry.register_tool(
        service_name=service_name,
        tool_name="echo",
        description="回显输入的文本",
        handler=echo_tool,
        parameters={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "要回显的消息",
                }
            },
            "required": ["message"],
        },
    )

    registry.register_tool(
        service_name=service_name,
        tool_name="add",
        description="计算两个数字的和",
        handler=add_tool,
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "第一个数字"},
                "b": {"type": "number", "description": "第二个数字"},
            },
            "required": ["a", "b"],
        },
    )

    # 3. 注册提示词
    registry.register_prompt(
        service_name=service_name,
        prompt_name="greeting",
        description="生成问候语提示词",
        handler=greeting_prompt,
        arguments=[
            {
                "name": "name",
                "description": "要问候的人的名字",
                "required": True,
            }
        ],
    )

    # 4. 注册资源
    registry.register_resource(
        service_name=service_name,
        resource_uri="example://info",
        description="服务信息",
        handler=info_resource,
        mime_type="application/json",
    )


# ===== 工具实现 =====


def echo_tool(message: str) -> str:
    """
    回显工具 - 返回输入的消息

    Args:
        message: 输入消息

    Returns:
        str: 回显的消息
    """
    logger.info(f"[Echo Tool] 收到消息: {message}")
    return f"Echo: {message}"


def add_tool(a: float, b: float) -> float:
    """
    加法工具 - 计算两个数字的和

    Args:
        a: 第一个数字
        b: 第二个数字

    Returns:
        float: 两数之和
    """
    result = a + b
    logger.info(f"[Add Tool] {a} + {b} = {result}")
    return result


# ===== 提示词实现 =====


def greeting_prompt(name: str) -> str:
    """
    问候提示词 - 生成个性化问候语

    Args:
        name: 要问候的人的名字

    Returns:
        str: 问候语提示词
    """
    prompt = f"""你是一个友好的AI助手。现在要向 {name} 问好。

请用热情、友好的语气生成一段问候语，包括:
1. 热情的打招呼
2. 询问对方今天过得如何
3. 表达乐意提供帮助

保持简洁、自然。"""

    logger.info(f"[Greeting Prompt] 为 {name} 生成问候提示词")
    return prompt


# ===== 资源实现 =====


def info_resource() -> dict:
    """
    信息资源 - 返回服务信息

    Returns:
        dict: 服务信息
    """
    logger.info("[Info Resource] 返回服务信息")
    return {
        "service": "example",
        "version": "1.0.0",
        "description": "这是一个示例MCP服务",
        "capabilities": {
            "tools": 2,
            "prompts": 1,
            "resources": 1,
        },
        "status": "active",
    }
