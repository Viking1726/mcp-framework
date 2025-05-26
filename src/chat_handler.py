"""
重构版聊天处理器 - 支持交互式工具调用
"""

import json
import logging
import asyncio
import re
import uuid
from typing import Dict, Any, List, AsyncIterator, Optional, Set, Tuple

from config import get_config
from llm_service_client import LLMServiceClient
from mcp_manager import MCPManager
from models import ChatCompletionRequest, ChatMessage
from session_manager import SessionManager
from tool_interactive import ToolResultStorage

logger = logging.getLogger(__name__)


class ChatHandler:
    """
    聊天处理器 - 处理与LLM的通信和工具调用
    """

    def __init__(self, llm_service_client: LLMServiceClient, mcp_manager: MCPManager, session_manager: SessionManager):
        self.llm_service = llm_service_client
        self.mcp_manager = mcp_manager
        self.session_manager = session_manager
        self.config = get_config()
        self.executed_tool_call_ids: Set[str] = set()
        self.continue_prompt = ChatMessage(
            role="user",
            content=getattr(self.config, "llm_continue_prompt",
                            "请根据工具执行的结果决定下一步操作。你可以继续使用工具，或者给出最终答案。")
        )
        # 工具结果存储
        self.tool_storage = ToolResultStorage()

    async def handle_chat_completion(
            self, request: ChatCompletionRequest, session_id: Optional[str] = None
    ) -> AsyncIterator[Any]:
        """处理聊天请求的主函数"""
        try:
            # 获取或创建会话
            session = self._get_or_create_session(session_id)

            # 增强请求
            self._enhance_request_with_tools(request)

            # 使用交互式处理
            async for chunk in self._handle_interactive_request(request):
                yield chunk

        except Exception as e:
            logger.error(f"处理请求时出错: {e}", exc_info=True)
            yield {
                "choices": [{
                    "delta": {"content": f"\n错误: {str(e)}\n"},
                    "index": 0,
                    "finish_reason": "error"
                }],
                "object": "chat.completion.chunk"
            }

    def _get_or_create_session(self, session_id: Optional[str]) -> Any:
        """获取或创建会话"""
        if session_id:
            return self.session_manager.get_session(session_id) or self.session_manager.create_session(session_id)
        else:
            return self.session_manager.create_session()

    def _enhance_request_with_tools(self, request: ChatCompletionRequest) -> None:
        """使用工具增强请求"""
        mcp_tools = self.mcp_manager.get_all_tools()
        if not mcp_tools:
            return

        # 记录工具信息
        logger.info(f"可用的工具: {len(mcp_tools)}")
        tool_names = [tool["function"]["name"]
                      for tool in mcp_tools
                      if "type" in tool and tool["type"] == "function" and "function" in tool]

        # 添加系统提示
        system_prompt = self._build_system_prompt(mcp_tools)
        request.messages = [ChatMessage(role="system", content=system_prompt)] + request.messages

        # 设置工具
        request.tools = mcp_tools
        request.tool_choice = getattr(self.config, "llm_tool_choice", "auto")

        # 记录模型信息
        logger.info(f"调用模型: {request.model or self.config.llm_service.model}")

    async def _handle_interactive_request(self, request: ChatCompletionRequest) -> AsyncIterator[Any]:
        """处理交互式请求"""
        # 设置超时保护
        response_timeout = getattr(self.config, "stream_response_timeout", 3600)  # 默认1小时

        try:
            # 使用超时包装整个处理过程
            async with asyncio.timeout(response_timeout):
                # 重置工具结果存储
                self.tool_storage = ToolResultStorage()
                self.executed_tool_call_ids = set()

                # 交互轮次
                interaction_round = 0
                max_rounds = getattr(self.config, "max_tool_interaction_rounds", 10)  # 最大交互轮次

                # 复制原始请求用于多轮交互
                current_request = ChatCompletionRequest(**request.dict())

                # 交互式工具调用循环
                while interaction_round < max_rounds:
                    interaction_round += 1
                    logger.info(f"开始工具交互轮次 {interaction_round}/{max_rounds}")

                    # 收集结果状态
                    state = {
                        "accumulated_content": "",
                        "tool_calls_data": [],  # 收集的工具调用数据
                        "finish_reason": None,  # 完成原因
                    }

                    # 第一阶段：获取模型响应和工具调用
                    async for chunk in self.llm_service.chat_completion(current_request, None):
                        # 处理响应块
                        self._process_response_chunk(chunk, state)

                        # 检查是否请求工具调用
                        if state["finish_reason"] == "tool_calls":
                            # 工具调用结束，跳出循环处理工具
                            break
                        elif state["finish_reason"]:
                            # 其他完成原因，直接返回结果给用户
                            yield chunk
                            return

                        # 继续传递响应块给用户
                        yield chunk

                    # 检查是否有工具调用
                    if state["finish_reason"] == "tool_calls" and state["tool_calls_data"]:
                        # 通知用户开始执行工具
                        yield {
                            "choices": [{
                                "delta": {"content": "\n\n[正在执行工具]\n\n"},
                                "index": 0,
                                "finish_reason": None
                            }],
                            "object": "chat.completion.chunk"
                        }

                        # 显示工具调用信息
                        tool_calls_json = json.dumps(state["tool_calls_data"], ensure_ascii=False, indent=2)
                        logger.info(f"收集到的工具调用: {tool_calls_json}")

                        yield {
                            "choices": [{
                                "delta": {"content": f"```json\n{tool_calls_json}\n```\n"},
                                "index": 0,
                                "finish_reason": None
                            }],
                            "object": "chat.completion.chunk"
                        }

                        # 执行工具调用
                        assistant_message = ChatMessage(
                            role="assistant",
                            content=state["accumulated_content"],
                            tool_calls=state["tool_calls_data"]
                        )

                        # 为下一轮交互准备消息
                        current_request.messages.append(assistant_message)

                        # 执行所有工具调用
                        for tool_call in state["tool_calls_data"]:
                            # 执行单个工具调用
                            result_message = await self._execute_single_tool(tool_call)

                            if result_message:
                                # 添加到消息列表
                                current_request.messages.append(result_message)

                                # try:
                                #     result_content = json.dumps(result_message.content, ensure_ascii=False, indent=2)
                                # except Exception as e:
                                #     logger.exception("解析工具结果失败")
                                #     result_content = str(result_message.content)

                                # 通知用户工具执行结果
                                function_name = tool_call.get("function", {}).get("name", "")

                                yield {
                                    "choices": [{
                                        "delta": {
                                            "content": f"\n\n[执行工具 {function_name} 结果]\n\n```json\n{result_message.content}\n```\n"
                                        },
                                        "index": 0,
                                        "finish_reason": None
                                    }],
                                    "object": "chat.completion.chunk"
                                }

                        # 添加工具结果上下文和提示
                        tool_context = self.tool_storage.get_context_for_model()
                        current_request.messages.append(ChatMessage(
                            role="user",
                            content=f"上面是工具执行的结果。\n\n{tool_context}\n\n请根据这些结果决定下一步操作。你可以继续调用其他工具，或者给出最终答案。"
                        ))

                        # 告知用户等待模型分析
                        yield {
                            "choices": [{
                                "delta": {"content": "\n\n[工具执行完成，等待分析...]\n\n"},
                                "index": 0,
                                "finish_reason": None
                            }],
                            "object": "chat.completion.chunk"
                        }
                    else:
                        # 没有工具调用，结束交互
                        logger.info("模型没有请求工具调用或已返回最终答案，交互结束")
                        return

        except asyncio.TimeoutError:
            logger.error(f"交互式响应超时: {response_timeout}秒内未完成")
            yield {
                "choices": [{
                    "delta": {"content": "\n\n[响应超时，工具交互可能未完成]\n"},
                    "index": 0,
                    "finish_reason": "timeout"
                }],
                "object": "chat.completion.chunk"
            }

    def _process_response_chunk(self, chunk: Dict[str, Any], state: Dict[str, Any]) -> None:
        """处理响应块，收集内容和工具调用"""
        if not isinstance(chunk, dict) or "choices" not in chunk:
            return

        # 获取主要的选择
        choice = chunk.get("choices", [{}])[0]

        # 检查完成原因
        if "finish_reason" in choice and choice["finish_reason"]:
            state["finish_reason"] = choice["finish_reason"]

        # 处理delta部分
        delta = choice.get("delta", {})

        # 收集内容
        if "content" in delta and delta["content"] is not None:
            state["accumulated_content"] += delta["content"]

        # 收集工具调用
        # if "tool_calls" in delta:
        #     # 处理每个工具调用
        #     for tool_call in delta.get("tool_calls", []):
        #         self._collect_tool_call(state["tool_calls_data"], tool_call)
        # 收集工具调用

        if "tool_calls" in delta:
            tool_calls = delta["tool_calls"]
            if tool_calls:  # 这样既检查了None也检查了空列表
                for tool_call in tool_calls:
                    self._collect_tool_call(state["tool_calls_data"], tool_call)

        # 处理整个消息（非增量情况）
        if "message" in choice:
            message = choice["message"]
            if "content" in message and message["content"] is not None:
                state["accumulated_content"] = message["content"]
            if "tool_calls" in message:
                state["tool_calls_data"] = message["tool_calls"]

    def _collect_tool_call(self, tool_calls_data: List[Dict[str, Any]],
                           tool_call: Dict[str, Any]) -> None:
        """收集流式工具调用数据"""
        index = tool_call.get("index", 0)

        # 确保索引位置有数据
        while len(tool_calls_data) <= index:
            tool_calls_data.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": "",
                    "arguments": ""
                }
            })

        # 更新基本数据
        if "id" in tool_call:
            tool_calls_data[index]["id"] = tool_call["id"]
        if "type" in tool_call:
            tool_calls_data[index]["type"] = tool_call["type"]

        # 更新函数数据
        if "function" in tool_call:
            self._update_function_data(tool_calls_data[index]["function"], tool_call["function"])

    def _update_function_data(self, current_function: Dict[str, Any],
                              function_data: Dict[str, Any]) -> None:
        """更新函数数据"""
        # 更新函数名 - 仅当当前函数名为空时
        if "name" in function_data:
            function_name = function_data.get("name", "")
            # 只有当当前函数名为空且新函数名有效时才更新
            if (not current_function.get("name")) and function_name and isinstance(function_name, str):
                current_function["name"] = function_name
                logger.info(f"收集到函数名称: {function_name}")
            elif function_name and isinstance(function_name, str):
                # 有效名称但不需要更新（已有名称）
                logger.debug(f"保留现有函数名称: {current_function.get('name')}")

        # 更新参数 - 始终累加
        if "arguments" in function_data:
            args = function_data.get("arguments", "")
            if not isinstance(args, str):
                args = str(args) if args is not None else ""
            current_function["arguments"] += args

    async def _execute_single_tool(self, tool_call: Dict[str, Any]) -> Optional[ChatMessage]:
        """执行单个工具调用"""
        tool_id = tool_call.get("id", f"call_{uuid.uuid4().hex[:8]}")

        # 避免重复执行
        if tool_id in self.executed_tool_call_ids:
            logger.info(f"工具调用 {tool_id} 已执行过，跳过")
            return None

        self.executed_tool_call_ids.add(tool_id)

        # 只处理函数类型工具
        if tool_call.get("type", "function") != "function":
            logger.warning(f"不支持的工具类型: {tool_call.get('type')}, 跳过")
            return None

        # 获取函数信息
        function = tool_call.get("function", {})
        function_name = function.get("name", "")
        arguments = function.get("arguments", "{}")

        # 检查函数名是否有效
        if not self._is_valid_function_name(function_name):
            logger.warning(f"工具调用缺少函数名称或名称无效: {repr(function_name)}")
            # 直接返回错误更加清晰
            return ChatMessage(
                role="tool",
                tool_call_id=tool_id,
                content=json.dumps({"error": "工具调用缺少有效的函数名称"})
            )

        logger.info(f"准备执行工具: {function_name} (ID: {tool_id})")

        # 处理参数中的引用
        processed_arguments = arguments
        try:
            # 解析和替换参数中的引用
            processed_arguments = self.tool_storage.resolve_references(arguments)
            if processed_arguments != arguments:
                logger.info(f"工具 {function_name} 的参数已处理依赖引用")
        except Exception as e:
            logger.error(f"处理工具参数引用时出错: {e}")

        # 执行工具并处理结果
        return await self._execute_tool_and_get_result(tool_id, function_name, processed_arguments)

    def _is_valid_function_name(self, name: Any) -> bool:
        """检查函数名是否有效"""
        return (name and isinstance(name, str) and name.strip() != "")

    async def _execute_tool_and_get_result(self, tool_id: str, function_name: str,
                                           arguments: Any) -> ChatMessage:
        """执行工具并获取结果"""
        try:
            # 解析参数
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    # 如果无法解析，尝试修复缺失的大括号问题
                    if not arguments.strip().startswith("{"):
                        arguments = "{" + arguments
                    if not arguments.strip().endswith("}"):
                        arguments = arguments + "}"
                    try:
                        arguments = json.loads(arguments)
                    except:
                        # 修复失败时使用空参数
                        arguments = {}

            logger.info(f"执行工具: {function_name}, 参数: {arguments}")

            try:
                # 执行工具调用
                result = await asyncio.wait_for(
                    self.mcp_manager.execute_tool(function_name, arguments),
                    timeout=300
                )
                logger.info(f"工具执行成功: {function_name}")

            except asyncio.TimeoutError:
                logger.error(f"工具 {function_name} 执行超时")
                result = {"error": "工具执行超时"}
            # 序列化结果
            serialized_result = self._serialize_result(result)
            tool_result = ChatMessage(
                role="tool",
                tool_call_id=tool_id,
                content=json.dumps(serialized_result, ensure_ascii=False)
            )

            # 记录工具结果
            logger.info(f"工具返回结果: {tool_result.content}")

            # 存储工具结果供后续引用
            try:
                self.tool_storage.store_result(function_name, serialized_result)
                logger.info(f"已存储工具 {function_name} 结果用于引用")
            except Exception as e:
                logger.error(f"存储工具结果失败: {e}")

            return tool_result

        except json.JSONDecodeError as e:
            logger.error(f"解析工具参数失败: {e}, 参数: {arguments}")
            return ChatMessage(
                role="tool",
                tool_call_id=tool_id,
                content=json.dumps({"error": f"参数解析失败: {str(e)}"})
            )

        except Exception as e:
            logger.error(f"工具 {function_name} 执行失败: {e}")
            return ChatMessage(
                role="tool",
                tool_call_id=tool_id,
                content=json.dumps({"error": str(e)})
            )

    def _build_system_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """
        构建系统提示以引导模型进行交互式工具调用
        """
        # 生成工具描述
        tool_descriptions = []
        for tool in tools:
            if "type" in tool and tool["type"] == "function":
                function = tool["function"]
                name = function.get("name", "")
                description = function.get("description", "")
                parameters = function.get("parameters", {})

                # 格式化参数信息
                param_info = ""
                if parameters and "properties" in parameters:
                    properties = parameters["properties"]
                    params_desc = []
                    for param_name, param_details in properties.items():
                        param_type = param_details.get("type", "any")
                        param_desc = param_details.get("description", "")
                        required = "required" in parameters and param_name in parameters["required"]
                        req_str = " (required)" if required else " (optional)"
                        params_desc.append(f"- {param_name}: {param_type}{req_str} - {param_desc}")

                    if params_desc:
                        param_info = "\n  参数:\n  " + "\n  ".join(params_desc)

                tool_descriptions.append(f"• {name}: {description}{param_info}")

        # 工具列表
        tool_list = "\n\n".join(tool_descriptions)

        # 构建交互式工具调用的系统提示，注意避免f-string中的大括号冲突
        path_example = '{"path": "[FILESYSTEM_LIST_ALLOWED_DIRECTORIES_RESULT]"}'

        # 构建系统提示
        return f"""## 交互式工具使用指南

你是一个强大的AI助手，能够使用工具来完成任务。你能够逐步使用工具，观察每个工具的执行结果，并据此决定下一步操作。

可用工具列表:
{tool_list}

交互式工具使用指南:

1. 分析任务和需求
   - 仔细理解用户的问题和目标
   - 确定解决问题需要哪些信息
   - 规划解决方案的步骤

2. 工具使用规则
   - 每次只使用一个工具
   - 查看工具执行结果后，再决定下一步操作
   - 可以使用 [TOOL_NAME_RESULT] 格式引用之前工具的执行结果
   - 例如: {path_example}

3. 问题解决流程
   - 从简单工具开始，获取基础信息
   - 观察每个工具执行的结果
   - 基于结果选择下一个最合适的工具
   - 持续调整策略直到完成任务

4. 何时使用工具
   - 当需要获取外部信息时
   - 当需要执行特定操作时
   - 当需要处理数据时

5. 何时给出最终答案
   - 当已获取足够信息解决用户问题时
   - 当所有必要的操作都已完成时
   - 当不再需要使用工具时

请记住，你的目标是帮助用户有效地解决问题，通过逐步使用工具获取信息并执行操作。"""

    def _serialize_result(self, obj: Any) -> Any:
        """序列化工具执行结果"""
        # 处理特殊对象
        if hasattr(obj, 'content') and isinstance(getattr(obj, 'content', None), str):
            return getattr(obj, 'content')

        # 递归处理复合类型
        if isinstance(obj, dict):
            return {k: self._serialize_result(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._serialize_result(item) for item in obj]

        # 处理基本类型
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj

        # 处理对象转换
        if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            return self._serialize_result(obj.to_dict())
        if hasattr(obj, 'asdict') and callable(getattr(obj, 'asdict')):
            return self._serialize_result(obj.asdict())

        # 尝试字符串转换
        try:
            return str(obj)
        except:
            return f"[不可序列化对象: {type(obj).__name__}]"
