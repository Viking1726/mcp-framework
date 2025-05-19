"""
改进版MCP管理器 - 工具名称处理更加灵活
"""

import asyncio
import json
import logging
import os
import sys
import inspect
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable, Awaitable, Set

# 导入MCP SDK组件
from mcp import ClientSession, StdioServerParameters, McpError
from mcp.client.stdio import stdio_client

# 从配置文件导入
from config import MCPServerConfig

logger = logging.getLogger(__name__)

@dataclass
class MCPServerConnection:
    """MCP服务器连接信息"""
    config: MCPServerConfig
    session: Optional[ClientSession] = None
    tools: List[Any] = field(default_factory=list)
    exit_stack: Optional[AsyncExitStack] = None
    is_connected: bool = False
    connection_attempt: int = 0

class MCPManager:
    """
    改进版MCP服务器管理器
    提供更灵活的工具名称处理
    """
    
    def __init__(self):
        self.servers: Dict[str, MCPServerConnection] = {}
        self.global_exit_stack = AsyncExitStack()
        # 添加工具名称映射缓存
        self._tool_name_cache: Dict[str, Tuple[str, str]] = {}
    
    async def start_server(self, config: MCPServerConfig) -> bool:
        """
        启动并连接到MCP服务器
        
        Args:
            config: MCP服务器配置
            
        Returns:
            bool: 连接是否成功
        """
        if not config.enabled:
            logger.info(f"服务器 {config.id} 未启用")
            return False
        
        if config.id in self.servers and self.servers[config.id].is_connected:
            logger.warning(f"服务器 {config.id} 已经在运行")
            return True
        
        # 创建/更新服务器记录
        server_conn = self.servers.get(config.id, MCPServerConnection(config=config))
        server_conn.connection_attempt += 1
        self.servers[config.id] = server_conn
        
        try:
            logger.info(f"启动 MCP 服务器: {config.id}")
            logger.info(f"命令: {[config.command] + config.args}")
            logger.info(f"超时设置: timeout={config.timeout}秒, init_timeout={config.init_timeout}秒")
            
            # 准备环境变量
            env = os.environ.copy()
            if config.env:
                env.update(config.env)
                logger.info(f"添加环境变量: {list(config.env.keys())}")
            
            # 创建一个新的退出栈 (AsyncExitStack) 用于管理此连接的资源
            server_conn.exit_stack = AsyncExitStack()
            await self.global_exit_stack.enter_async_context(server_conn.exit_stack)
            
            # 创建服务器参数
            server_params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=env
            )
            
            # 创建stdio连接
            try:
                logger.info(f"建立连接到 {config.id} 服务器")
                stdio_transport = await server_conn.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                stdio, write = stdio_transport
                
                # 创建客户端会话
                session = await server_conn.exit_stack.enter_async_context(
                    ClientSession(stdio, write)
                )
                server_conn.session = session
                
                # 执行初始化，添加超时处理
                init_timeout = config.init_timeout or 30
                logger.info(f"初始化 {config.id} 服务器 (超时: {init_timeout}秒)")
                
                # 使用外部超时来包装initialize调用
                try:
                    await asyncio.wait_for(
                        session.initialize(),
                        timeout=init_timeout
                    )
                    logger.info(f"服务器 {config.id} 初始化成功")
                except asyncio.TimeoutError:
                    logger.error(f"服务器 {config.id} 初始化超时")
                    raise  # 重新抛出，在外部处理
                
                # 获取可用工具，添加超时处理
                tools_timeout = config.timeout or 30
                try:
                    tools_response = await asyncio.wait_for(
                        session.list_tools(),
                        timeout=tools_timeout
                    )
                    server_conn.tools = tools_response.tools
                    
                    tool_names = [tool.name for tool in server_conn.tools]
                    logger.info(f"服务器 {config.id} 提供的工具: {tool_names}")
                    
                    # 清除旧的工具名称缓存
                    self._tool_name_cache.clear()
                    
                except asyncio.TimeoutError:
                    logger.error(f"获取服务器 {config.id} 工具列表超时")
                    raise  # 重新抛出，在外部处理
                
                server_conn.is_connected = True
                server_conn.connection_attempt = 0  # 重置尝试计数
                
                return True
                
            except asyncio.TimeoutError as e:
                logger.error(f"连接到服务器 {config.id} 超时: {e}")
                await self._cleanup_server_connection(config.id)
                
                # 如果尝试次数小于3，允许后续重试
                if server_conn.connection_attempt < 3:
                    logger.info(f"服务器 {config.id} 将在下次请求时重试连接")
                    return False
                else:
                    logger.error(f"服务器 {config.id} 连接失败达到最大尝试次数")
                    return False
                
            except McpError as e:
                logger.error(f"服务器 {config.id} 初始化失败: {e}")
                await self._cleanup_server_connection(config.id)
                return False
                
            except Exception as e:
                logger.error(f"连接到服务器 {config.id} 时发生未知错误: {e}", exc_info=True)
                await self._cleanup_server_connection(config.id)
                return False
                
        except Exception as e:
            logger.error(f"启动服务器 {config.id} 失败: {e}", exc_info=True)
            await self._cleanup_server_connection(config.id)
            return False
    
    async def _cleanup_server_connection(self, server_id: str):
        """清理服务器连接资源"""
        if server_id not in self.servers:
            return
        
        server_conn = self.servers[server_id]
        if server_conn.exit_stack:
            try:
                await server_conn.exit_stack.aclose()
            except Exception as e:
                logger.error(f"清理服务器 {server_id} 资源时出错: {e}")
            
            server_conn.exit_stack = None
        
        server_conn.session = None
        server_conn.is_connected = False
        server_conn.tools = []
    
    async def stop_server(self, server_id: str):
        """停止服务器连接"""
        await self._cleanup_server_connection(server_id)
        
        if server_id in self.servers:
            del self.servers[server_id]
            logger.info(f"已停止服务器: {server_id}")
    
    async def stop_all_servers(self):
        """停止所有服务器连接"""
        for server_id in list(self.servers.keys()):
            await self.stop_server(server_id)
        
        # 清理全局资源
        await self.global_exit_stack.aclose()
        self.global_exit_stack = AsyncExitStack()
        # 清除工具名称缓存
        self._tool_name_cache.clear()
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """
        获取所有工具（OpenAI SDK工具格式）
        
        Returns:
            List[Dict]: OpenAI工具格式的工具列表
        """
        tools = []
        tool_name_conflicts = set()  # 跟踪命名冲突
        
        # 第一遍扫描，找出所有潜在的命名冲突
        for server_id, server_conn in self.servers.items():
            if server_conn.is_connected and server_conn.tools:
                for tool in server_conn.tools:
                    # 检查是否已存在同名工具
                    if tool.name in self._tool_name_cache:
                        tool_name_conflicts.add(tool.name)
        
        # 第二遍扫描，创建工具并更新缓存
        for server_id, server_conn in self.servers.items():
            if server_conn.is_connected and server_conn.tools:
                for tool in server_conn.tools:
                    # 创建带有服务器ID前缀的工具名
                    prefixed_name = f"{server_id}_{tool.name}"
                    
                    # 创建OpenAI SDK兼容的工具定义
                    function = {
                        "type": "function",
                        "function": {
                            "name": prefixed_name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    }
                    tools.append(function)
                    
                    # 更新工具名称缓存
                    self._tool_name_cache[prefixed_name] = (server_id, tool.name)
                    
                    # 如果没有命名冲突，也添加不带前缀的工具名
                    if tool.name not in tool_name_conflicts:
                        self._tool_name_cache[tool.name] = (server_id, tool.name)
                        logger.debug(f"工具 {tool.name} 可以通过无前缀名称访问")
        
        return tools
    
    async def ensure_server_connected(self, server_id: str) -> bool:
        """
        确保服务器已连接，如果未连接则尝试重新连接
        
        Args:
            server_id: 服务器ID
            
        Returns:
            bool: 服务器是否连接成功
        """
        if server_id not in self.servers:
            logger.error(f"服务器 {server_id} 不存在")
            return False
        
        server_conn = self.servers[server_id]
        
        # 如果已连接，直接返回成功
        if server_conn.is_connected and server_conn.session:
            return True
        
        # 如果未连接但尝试次数过多，返回失败
        if server_conn.connection_attempt >= 3:
            logger.error(f"服务器 {server_id} 连接失败次数过多，不再尝试")
            return False
        
        # 尝试重新连接
        logger.info(f"服务器 {server_id} 未连接，尝试重新连接")
        return await self.start_server(server_conn.config)

    def resolve_tool_name(self, tool_name: str) -> Optional[Tuple[str, str]]:
        """
        解析工具名称，返回 (server_id, actual_tool_name)
        支持直接使用工具名称而不需要服务器前缀

        Args:
            tool_name: 工具名称（可以带服务器前缀，也可以不带）

        Returns:
            Optional[Tuple[str, str]]: (server_id, actual_tool_name) 或 None
        """
        # 安全检查：处理空或非字符串输入
        if not tool_name or not isinstance(tool_name, str):
            logger.error(f"无效的工具名称: {tool_name}")
            return None

        # 首先尝试从缓存中获取
        if tool_name in self._tool_name_cache:
            return self._tool_name_cache[tool_name]

        # 如果包含下划线，尝试解析为服务器ID和工具名
        if "_" in tool_name:
            parts = tool_name.split("_", 1)
            if len(parts) == 2:
                server_id, actual_tool_name = parts

                # 检查服务器是否存在
                if server_id in self.servers:
                    # 检查工具是否存在
                    server_conn = self.servers[server_id]
                    if server_conn.is_connected and server_conn.tools:
                        for tool in server_conn.tools:
                            if tool.name == actual_tool_name:
                                # 找到了匹配的工具，缓存并返回
                                self._tool_name_cache[tool_name] = (server_id, actual_tool_name)
                                return server_id, actual_tool_name

        # 尝试寻找所有服务器中可能匹配的工具
        for server_id, server_conn in self.servers.items():
            if server_conn.is_connected and server_conn.tools:
                for tool in server_conn.tools:
                    if tool.name == tool_name:
                        # 找到不带前缀的工具名称
                        result = (server_id, tool.name)
                        self._tool_name_cache[tool_name] = result
                        return result

        # 没有找到匹配的工具
        return None

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        执行工具调用 - 改进版本支持更灵活的工具名称解析

        Args:
            tool_name: 工具名称 (可以是 "server_id_tool_name" 或直接是 "tool_name")
            arguments: 工具参数

        Returns:
            Any: 工具执行结果

        Raises:
            ValueError: 无效的工具名称或服务器不存在
            Exception: 工具调用失败
        """
        # 安全检查：验证工具名是否为有效字符串
        if not tool_name or not isinstance(tool_name, str) or tool_name.strip() == "":
            error_msg = f"工具名称无效或为空: {repr(tool_name)}"
            logger.error(error_msg)
            return {"error": error_msg}  # 返回错误对象而不是抛出异常

        # 解析工具名称
        resolved = self.resolve_tool_name(tool_name)
        if not resolved:
            # 输出所有可用的工具名称用于调试
            available_tools = []
            for server_id, server_conn in self.servers.items():
                if server_conn.is_connected and server_conn.tools:
                    for tool in server_conn.tools:
                        available_tools.append(f"{server_id}_{tool.name}")

            error_msg = f"无法解析工具名: {tool_name}。可用工具: {available_tools}"
            logger.error(error_msg)
            return {"error": error_msg}  # 返回错误对象而不是抛出异常
        
        server_id, actual_tool_name = resolved
        logger.info(f"解析工具名称: {tool_name} -> 服务器: {server_id}, 工具: {actual_tool_name}")
        
        # 确保服务器已连接
        if not await self.ensure_server_connected(server_id):
            raise ValueError(f"服务器 {server_id} 未连接")
        
        server_conn = self.servers[server_id]
        
        # 检查工具是否可用
        available_tool_names = [tool.name for tool in server_conn.tools]
        if actual_tool_name not in available_tool_names:
            raise ValueError(f"工具 {actual_tool_name} 在服务器 {server_id} 上不可用。可用工具: {available_tool_names}")
        
        # 执行工具调用（带重试）
        max_retries = 2
        retry_delay = 1.0
        last_error = None
        timeout = server_conn.config.timeout or 30
        
        for attempt in range(1, max_retries + 1):
            try:
                # 检查call_tool方法是否接受额外参数
                call_tool_params = len(inspect.signature(server_conn.session.call_tool).parameters)
                
                # 调用工具并获取结果，添加超时处理
                if call_tool_params > 2:  # 如果接受超过2个参数(self, name, arguments)
                    try:
                        # 尝试导入CancellationToken，如果存在的话
                        from mcp.types import CancellationToken
                        result = await asyncio.wait_for(
                            server_conn.session.call_tool(actual_tool_name, arguments, CancellationToken()),
                            timeout=timeout
                        )
                    except ImportError:
                        # 如果CancellationToken不存在，使用None
                        result = await asyncio.wait_for(
                            server_conn.session.call_tool(actual_tool_name, arguments, None),
                            timeout=timeout
                        )
                else:
                    # 不需要额外参数
                    result = await asyncio.wait_for(
                        server_conn.session.call_tool(actual_tool_name, arguments),
                        timeout=timeout
                    )
                logger.info("result.content")
                logger.info(result.content)
                return result.content
                
            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning(f"工具调用 {tool_name} 超时，尝试 {attempt}/{max_retries}")
                
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                
            except McpError as e:
                # 服务器报告的错误，通常不需要重试
                raise Exception(f"工具调用 {tool_name} 失败: {e}")
                
            except Exception as e:
                last_error = e
                logger.warning(f"工具调用 {tool_name} 失败，尝试 {attempt}/{max_retries}: {e}")
                
                if attempt < max_retries:
                    # 如果是连接错误，尝试重新连接
                    if "connection" in str(e).lower() or "broken pipe" in str(e).lower():
                        await self.stop_server(server_id)
                        if not await self.start_server(server_conn.config):
                            raise Exception(f"工具调用 {tool_name} 失败: 无法重新连接到服务器")
                        
                        # 重置服务器连接对象，因为旧的已经被清理
                        server_conn = self.servers[server_id]
                    else:
                        # 其他错误只是等待后重试
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
        
        # 如果达到重试次数限制，抛出最后一个错误
        if last_error:
            raise Exception(f"工具调用 {tool_name} 失败，已达到最大重试次数: {last_error}")
        else:
            raise Exception(f"工具调用 {tool_name} 失败: 未知错误")

    async def ping_server(self, server_id: str) -> bool:
        """
        检查服务器是否响应
        
        Args:
            server_id: 服务器ID
            
        Returns:
            bool: 服务器是否响应
        """
        if server_id not in self.servers:
            return False
        
        server_conn = self.servers[server_id]
        if not server_conn.is_connected or not server_conn.session:
            return False
        
        try:
            # 使用list_tools作为ping操作，添加较短的超时
            await asyncio.wait_for(
                server_conn.session.list_tools(),
                timeout=5.0
            )
            return True
        except Exception:
            return False
    
    async def refresh_tools(self, server_id: str) -> List[Any]:
        """
        刷新服务器上的工具列表
        
        Args:
            server_id: 服务器ID
            
        Returns:
            List[Any]: 刷新后的工具列表
        """
        if not await self.ensure_server_connected(server_id):
            return []
        
        server_conn = self.servers[server_id]
        timeout = server_conn.config.timeout or 30
        
        try:
            tools_response = await asyncio.wait_for(
                server_conn.session.list_tools(),
                timeout=timeout
            )
            server_conn.tools = tools_response.tools
            
            # 清除工具名称缓存强制重建
            self._tool_name_cache.clear()
            
            return server_conn.tools
        except Exception as e:
            logger.error(f"刷新服务器 {server_id} 工具列表失败: {e}")
            return []
