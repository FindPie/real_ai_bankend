# Real AI MCP 服务中心

基于 FastMCP 的 MCP（Model Context Protocol）服务中心，提供统一的服务注册和调用机制。

## 目录结构

```
mcp_center/
├── __init__.py           # 包初始化
├── config.py             # 配置文件
├── registry.py           # 服务注册中心
├── server.py             # MCP 服务器
├── services/             # 服务目录
│   ├── __init__.py
│   └── example_service.py  # 示例服务
└── README.md             # 本文档

run_mcp_server.py         # 启动脚本
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行服务器

```bash
python run_mcp_server.py
```

### 3. 配置环境变量（可选）

```bash
export MCP_SERVICE_NAME="My MCP Server"
export MCP_SERVICE_VERSION="1.0.0"
export MCP_HOST="localhost"
export MCP_PORT="8100"
export MCP_LOG_LEVEL="INFO"
export MCP_AUTO_DISCOVER_SERVICES="true"
```

## 如何添加新服务

### 1. 创建服务文件

在 `mcp_center/services/` 目录下创建新的 Python 文件，例如 `my_service.py`：

```python
"""
我的自定义服务
"""
from loguru import logger
from mcp_center.registry import registry


def initialize():
    """初始化服务"""
    service_name = "my_service"

    # 1. 注册服务
    registry.register_service(
        name=service_name,
        description="我的自定义MCP服务",
        version="1.0.0",
    )

    # 2. 注册工具
    registry.register_tool(
        service_name=service_name,
        tool_name="my_tool",
        description="我的工具描述",
        handler=my_tool_handler,
        parameters={
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "参数1描述",
                }
            },
            "required": ["param1"],
        },
    )

    # 3. 注册提示词（可选）
    registry.register_prompt(
        service_name=service_name,
        prompt_name="my_prompt",
        description="我的提示词描述",
        handler=my_prompt_handler,
        arguments=[
            {
                "name": "arg1",
                "description": "参数1描述",
                "required": True,
            }
        ],
    )

    # 4. 注册资源（可选）
    registry.register_resource(
        service_name=service_name,
        resource_uri="my_service://resource",
        description="我的资源描述",
        handler=my_resource_handler,
        mime_type="application/json",
    )


# 工具实现
def my_tool_handler(param1: str) -> str:
    """工具处理函数"""
    logger.info(f"执行工具: param1={param1}")
    return f"处理结果: {param1}"


# 提示词实现
def my_prompt_handler(arg1: str) -> str:
    """提示词处理函数"""
    logger.info(f"生成提示词: arg1={arg1}")
    return f"提示词内容..."


# 资源实现
def my_resource_handler() -> dict:
    """资源处理函数"""
    logger.info("返回资源")
    return {
        "status": "ok",
        "data": "资源数据"
    }
```

### 2. 重启服务器

服务会自动被发现和加载（如果 `auto_discover_services=True`）。

## MCP 概念

### Tools（工具）

工具是可以被 AI 模型调用的函数，用于执行特定任务。

- **参数定义**: 使用 JSON Schema 格式定义工具参数
- **处理函数**: 实现工具的具体逻辑
- **返回值**: 可以是任何可序列化的 Python 对象

### Prompts（提示词）

提示词模板用于生成特定格式的提示内容。

- **参数定义**: 定义提示词需要的参数
- **处理函数**: 生成最终的提示词文本
- **返回值**: 通常是字符串类型的提示词内容

### Resources（资源）

资源是可以被访问的数据或内容。

- **URI**: 唯一标识资源的路径
- **MIME类型**: 资源的内容类型
- **处理函数**: 返回资源内容

## 配置说明

### MCPSettings 配置项

| 配置项 | 类型 | 默认值 | 说明 |
|-------|------|--------|------|
| service_name | str | "Real AI MCP Server" | 服务名称 |
| service_version | str | "1.0.0" | 服务版本 |
| service_description | str | "Real AI MCP服务中心..." | 服务描述 |
| host | str | "localhost" | 服务器主机 |
| port | int | 8100 | 服务器端口 |
| log_level | str | "INFO" | 日志级别 |
| auto_discover_services | bool | True | 自动发现服务 |

所有配置项都可以通过环境变量覆盖，前缀为 `MCP_`。

## 注册中心 API

### registry.register_service()

注册一个新服务。

```python
registry.register_service(
    name="service_name",
    description="服务描述",
    version="1.0.0"
)
```

### registry.register_tool()

注册一个工具。

```python
registry.register_tool(
    service_name="service_name",
    tool_name="tool_name",
    description="工具描述",
    handler=handler_function,
    parameters={...}  # JSON Schema
)
```

### registry.register_prompt()

注册一个提示词。

```python
registry.register_prompt(
    service_name="service_name",
    prompt_name="prompt_name",
    description="提示词描述",
    handler=handler_function,
    arguments=[...]
)
```

### registry.register_resource()

注册一个资源。

```python
registry.register_resource(
    service_name="service_name",
    resource_uri="scheme://path",
    description="资源描述",
    handler=handler_function,
    mime_type="application/json"
)
```

## 开发建议

1. **模块化设计**: 每个服务应该是独立的 Python 文件
2. **统一命名**: 使用清晰、描述性的服务名和工具名
3. **完善文档**: 为工具和提示词提供详细的描述
4. **错误处理**: 在处理函数中添加适当的异常处理
5. **日志记录**: 使用 loguru 记录关键操作
6. **类型注解**: 为函数参数添加类型提示

## 示例服务

查看 `services/example_service.py` 获取完整示例。

## 许可证

MIT
