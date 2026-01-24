"""
MCP服务注册中心
"""
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class MCPServiceInfo:
    """MCP服务信息"""

    name: str
    description: str
    version: str = "1.0.0"
    tools: List[Dict[str, Any]] = field(default_factory=list)
    prompts: List[Dict[str, Any]] = field(default_factory=list)
    resources: List[Dict[str, Any]] = field(default_factory=list)
    enabled: bool = True


class MCPRegistry:
    """MCP服务注册中心"""

    def __init__(self):
        self._services: Dict[str, MCPServiceInfo] = {}
        self._tool_handlers: Dict[str, Callable] = {}
        self._prompt_handlers: Dict[str, Callable] = {}
        self._resource_handlers: Dict[str, Callable] = {}

    def register_service(
        self,
        name: str,
        description: str,
        version: str = "1.0.0",
    ) -> MCPServiceInfo:
        """
        注册一个新的MCP服务

        Args:
            name: 服务名称
            description: 服务描述
            version: 服务版本

        Returns:
            MCPServiceInfo: 服务信息对象
        """
        if name in self._services:
            logger.warning(f"服务 '{name}' 已存在，将被覆盖")

        service_info = MCPServiceInfo(
            name=name, description=description, version=version
        )
        self._services[name] = service_info
        logger.info(f"✓ 注册服务: {name} (v{version})")
        return service_info

    def register_tool(
        self,
        service_name: str,
        tool_name: str,
        description: str,
        handler: Callable,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        注册一个工具到指定服务

        Args:
            service_name: 服务名称
            tool_name: 工具名称
            description: 工具描述
            handler: 工具处理函数
            parameters: 工具参数定义
        """
        if service_name not in self._services:
            raise ValueError(f"服务 '{service_name}' 未注册")

        tool_info = {
            "name": tool_name,
            "description": description,
            "parameters": parameters or {},
        }

        self._services[service_name].tools.append(tool_info)
        handler_key = f"{service_name}.{tool_name}"
        self._tool_handlers[handler_key] = handler
        logger.info(f"  ✓ 注册工具: {handler_key}")

    def register_prompt(
        self,
        service_name: str,
        prompt_name: str,
        description: str,
        handler: Callable,
        arguments: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        注册一个提示词到指定服务

        Args:
            service_name: 服务名称
            prompt_name: 提示词名称
            description: 提示词描述
            handler: 提示词处理函数
            arguments: 提示词参数定义
        """
        if service_name not in self._services:
            raise ValueError(f"服务 '{service_name}' 未注册")

        prompt_info = {
            "name": prompt_name,
            "description": description,
            "arguments": arguments or [],
        }

        self._services[service_name].prompts.append(prompt_info)
        handler_key = f"{service_name}.{prompt_name}"
        self._prompt_handlers[handler_key] = handler
        logger.info(f"  ✓ 注册提示词: {handler_key}")

    def register_resource(
        self,
        service_name: str,
        resource_uri: str,
        description: str,
        handler: Callable,
        mime_type: str = "text/plain",
    ):
        """
        注册一个资源到指定服务

        Args:
            service_name: 服务名称
            resource_uri: 资源URI
            description: 资源描述
            handler: 资源处理函数
            mime_type: 资源MIME类型
        """
        if service_name not in self._services:
            raise ValueError(f"服务 '{service_name}' 未注册")

        resource_info = {
            "uri": resource_uri,
            "description": description,
            "mime_type": mime_type,
        }

        self._services[service_name].resources.append(resource_info)
        self._resource_handlers[resource_uri] = handler
        logger.info(f"  ✓ 注册资源: {resource_uri}")

    def get_service(self, name: str) -> Optional[MCPServiceInfo]:
        """获取服务信息"""
        return self._services.get(name)

    def list_services(self) -> List[MCPServiceInfo]:
        """列出所有已注册的服务"""
        return list(self._services.values())

    def get_tool_handler(self, service_name: str, tool_name: str) -> Optional[Callable]:
        """获取工具处理函数"""
        handler_key = f"{service_name}.{tool_name}"
        return self._tool_handlers.get(handler_key)

    def get_prompt_handler(
        self, service_name: str, prompt_name: str
    ) -> Optional[Callable]:
        """获取提示词处理函数"""
        handler_key = f"{service_name}.{prompt_name}"
        return self._prompt_handlers.get(handler_key)

    def get_resource_handler(self, resource_uri: str) -> Optional[Callable]:
        """获取资源处理函数"""
        return self._resource_handlers.get(resource_uri)

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """获取所有工具列表"""
        tools = []
        for service_name, service in self._services.items():
            if service.enabled:
                for tool in service.tools:
                    tools.append(
                        {
                            "service": service_name,
                            "name": tool["name"],
                            "description": tool["description"],
                            "parameters": tool["parameters"],
                        }
                    )
        return tools

    def get_all_prompts(self) -> List[Dict[str, Any]]:
        """获取所有提示词列表"""
        prompts = []
        for service_name, service in self._services.items():
            if service.enabled:
                for prompt in service.prompts:
                    prompts.append(
                        {
                            "service": service_name,
                            "name": prompt["name"],
                            "description": prompt["description"],
                            "arguments": prompt["arguments"],
                        }
                    )
        return prompts

    def get_all_resources(self) -> List[Dict[str, Any]]:
        """获取所有资源列表"""
        resources = []
        for service_name, service in self._services.items():
            if service.enabled:
                for resource in service.resources:
                    resources.append(
                        {
                            "service": service_name,
                            "uri": resource["uri"],
                            "description": resource["description"],
                            "mime_type": resource["mime_type"],
                        }
                    )
        return resources


# 全局注册中心实例
registry = MCPRegistry()
