import logging
import time
import json
import re
import aiohttp
from typing import Dict, Any, List, Optional, AsyncIterator
from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletion
from models import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk, ModelListResponse, ModelObject
from config import get_config

logger = logging.getLogger(__name__)

class LLMServiceClient:
    """使用OpenAI官方SDK的增强客户端
    
    支持的服务包括但不限于:
    - OpenAI API
    - Azure OpenAI API
    - LM Studio
    - Ollama
    - Qwen3
    - 任何OpenAI API兼容的服务(通过base_url配置)
    """
    
    def __init__(self):
        self.config = get_config().llm_service
        self.base_url = self.config.base_url
        
        # 检测服务类型
        self.service_type = self._detect_service_type()
        
        # 初始化客户端
        self._initialize_client()
    
    def _detect_service_type(self) -> str:
        """基于URL和配置自动检测服务类型"""
        # 如果指定了服务类型，直接使用
        if hasattr(self.config, 'service_type') and self.config.service_type:
            return self.config.service_type
            
        # 否则从 URL 或其他配置中检测
        base_url = self.base_url.lower()
        
        if "ollama" in base_url or "11434" in base_url:
            return "ollama"
        elif "anthropic" in base_url:
            return "anthropic"
        elif "lmstudio" in base_url or "127.0.0.1:1234" in base_url or "localhost:1234" in base_url:
            return "lmstudio"
        elif "chat.api.qwen" in base_url or "dashscope" in base_url:
            return "qwen"
        elif "azure" in base_url:
            return "azure"
        elif "localhost" in base_url or "127.0.0.1" in base_url:
            return "local"
        else:
            return "openai"  # 默认使用 OpenAI
    
    def _initialize_client(self):
        """根据服务类型初始化客户端"""
        headers = {}
        api_key = getattr(self.config, 'api_key', 'dummy-key')
        
        # 根据服务类型设置特殊头
        if self.service_type == "ollama":
            api_key = "ollama"
            headers["Authorization"] = f"Bearer {api_key}"
            
        elif self.service_type == "anthropic":
            headers["x-api-key"] = api_key
            if hasattr(self.config, "extra_body") and self.config.extra_body:
                if "enable_thinking" in self.config.extra_body:
                    headers["anthropic-beta"] = "thinking-in-llms-v0"
                    
        elif self.service_type == "qwen":
            headers["Authorization"] = f"Bearer {api_key}"
            if hasattr(self.config, "extra_body") and self.config.extra_body:
                if "enable_thinking" in self.config.extra_body:
                    headers["X-DashScope-Beta"] = "thinking-in-llms-v1"
                    
        elif self.service_type == "azure":
            headers["api-key"] = api_key
            
        elif self.service_type in ["lmstudio", "local"]:
            # 对于本地服务，通常不需要认证
            api_key = "dummy-key"
        
        # 创建客户端
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=getattr(self.config, 'timeout', 300),
            max_retries=getattr(self.config, 'max_retries', 2),
            default_headers=headers if headers else None
        )
        
        # 记录日志
        logger.info(f"初始化LLM客户端: {self.service_type} ({self.base_url})")
    
    async def chat_completion(
        self, 
        request: ChatCompletionRequest,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncIterator[Any]:
        """
        调用聊天完成 API
        
        Args:
            request: ChatCompletionRequest 对象
            tools: MCP 工具列表（转换为 OpenAI 函数格式）
        
        Yields:
            如果是流式响应，yield ChatCompletionChunk
            如果是非流式响应，yield ChatCompletionResponse
        """
        # 准备请求数据
        data = request.dict(exclude_unset=True)
        
        # 特殊处理本地模型和Ollama
        if self.service_type in ["ollama", "lmstudio", "local"]:
            # 修复：不使用return，而是使用异步for循环
            async for item in self._chat_completion_with_http(request, tools):
                yield item
            return  # 提前返回，不执行后面的代码
        
        # 使用配置中的模型
        model = data.get("model") or self.config.model
        
        # 处理参数 - 始终使用流式模式
        kwargs = {
            "model": model,
            "messages": data.get("messages", []),
            "temperature": data.get("temperature", self.config.temperature),
            "stream": True  # 始终使用流式模式
        }
        
        # 添加可选参数
        if data.get("max_tokens") is not None:
            kwargs["max_tokens"] = data.get("max_tokens")
        elif hasattr(self.config, "max_tokens"):
            kwargs["max_tokens"] = self.config.max_tokens
            
        if data.get("top_p") is not None:
            kwargs["top_p"] = data.get("top_p")
            
        if data.get("frequency_penalty") is not None:
            kwargs["frequency_penalty"] = data.get("frequency_penalty")
            
        if data.get("presence_penalty") is not None:
            kwargs["presence_penalty"] = data.get("presence_penalty")
            
        if data.get("stop") is not None:
            kwargs["stop"] = data.get("stop")
        
        # 处理工具 - 首先直接使用请求中的tools字段（如果存在）
        if data.get("tools") and len(data.get("tools", [])) > 0:
            kwargs["tools"] = data.get("tools")
            kwargs["tool_choice"] = data.get("tool_choice", "auto")
        # 如果没有tools字段，但有外部输入的tools参数
        elif tools and len(tools) > 0:
            kwargs["tools"] = tools
            # 处理兼容性：将旧的function_call转换为tool_choice
            if data.get("function_call"):
                if data["function_call"] == "auto":
                    kwargs["tool_choice"] = "auto"
                elif data["function_call"] == "none":
                    kwargs["tool_choice"] = "none"
                else:
                    # 特定函数调用
                    func_data = data["function_call"]
                    if isinstance(func_data, str) and func_data != "auto" and func_data != "none":
                        # 如果是函数名称字符串
                        kwargs["tool_choice"] = {"type": "function", "function": {"name": func_data}}
                    elif isinstance(func_data, dict) and "name" in func_data:
                        # 如果是包含name的对象
                        kwargs["tool_choice"] = {"type": "function", "function": {"name": func_data["name"]}}
                    else:
                        kwargs["tool_choice"] = "auto"
            else:
                kwargs["tool_choice"] = "auto"
        # 如果请求中有functions字段（兼容旧格式）
        elif data.get("functions") and len(data.get("functions", [])) > 0:
            # 将旧的functions格式转换为新的tools格式
            functions = data.get("functions", [])
            converted_tools = []
            for func in functions:
                if isinstance(func, dict):
                    tool = {
                        "type": "function",
                        "function": {
                            "name": func.get("name", ""),
                            "description": func.get("description", ""),
                            "parameters": func.get("parameters", {})
                        }
                    }
                    converted_tools.append(tool)
            
            if converted_tools:
                kwargs["tools"] = converted_tools
                
                # 处理function_call转换为tool_choice
                if data.get("function_call"):
                    if data["function_call"] == "auto":
                        kwargs["tool_choice"] = "auto"
                    elif data["function_call"] == "none":
                        kwargs["tool_choice"] = "none"
                    else:
                        # 特定函数调用
                        func_data = data["function_call"]
                        if isinstance(func_data, str) and func_data != "auto" and func_data != "none":
                            # 如果是函数名称字符串
                            kwargs["tool_choice"] = {"type": "function", "function": {"name": func_data}}
                        elif isinstance(func_data, dict) and "name" in func_data:
                            # 如果是包含name的对象
                            kwargs["tool_choice"] = {"type": "function", "function": {"name": func_data["name"]}}
                        else:
                            kwargs["tool_choice"] = "auto"
                else:
                    kwargs["tool_choice"] = "auto"
        
        # 处理extra_body配置（如果存在）
        if hasattr(self.config, "extra_body") and self.config.extra_body:
            # 对于不同服务的特殊参数处理
            if self.service_type == "anthropic":
                if "thinking_budget" in self.config.extra_body:
                    kwargs["thinking_budget"] = self.config.extra_body.get("thinking_budget")
            elif self.service_type == "azure":
                # Azure需要api-version参数
                if "api-version" in self.config.extra_body:
                    kwargs["extra_query"] = {"api-version": self.config.extra_body.get("api-version")}
                    
            # 添加其他extra_body中的参数
            for key, value in self.config.extra_body.items():
                if key not in ["enable_thinking", "thinking_budget", "api-version"] and key not in kwargs:
                    kwargs[key] = value
            
            logger.debug(f"应用额外参数: {[k for k in kwargs if k not in ['model', 'messages', 'temperature', 'stream']]}")

        try:
            # 始终使用流式响应
            stream = await self.client.chat.completions.create(**kwargs)
            async for chunk in stream:
                # 转换为OpenAI兼容格式
                chunk_dict = chunk.model_dump()
                # 打印原始响应结果
                # todo 原始打印单独记录
                # logger.info(f"模型原始响应: {json.dumps(chunk_dict, ensure_ascii=False)}")
                yield chunk_dict
        
        except OpenAIError as e:
            logger.error(f"OpenAI API 请求失败: {e}")
            raise
        except Exception as e:
            logger.error(f"未知错误: {e}")
            raise
    
    async def _chat_completion_with_http(self, request: ChatCompletionRequest, tools: Optional[List[Dict[str, Any]]] = None) -> AsyncIterator[Any]:
        """使用HTTP请求处理特殊服务（比如Ollama和LM Studio）"""
        # 准备请求数据
        data = request.dict(exclude_unset=True)
        
        # 处理工具 - 修复工具转换逻辑
        if tools and len(tools) > 0:
            # 根据服务类型选择合适的格式
            if self.service_type in ["ollama", "lmstudio", "local"]:
                # 将新工具格式转换回旧格式
                functions = []
                for tool in tools:
                    if "type" in tool and tool["type"] == "function" and "function" in tool:
                        func = tool["function"]
                        functions.append({
                            "name": func.get("name", ""),
                            "description": func.get("description", ""),
                            "parameters": func.get("parameters", {})
                        })
                    else:
                        # 原样保留不符合格式的工具
                        functions.append(tool)
                
                data["functions"] = functions
                if not data.get("function_call"):
                    data["function_call"] = "auto"
            else:
                # 保持OpenAI新格式
                data["tools"] = tools
                # 处理tool_choice
                if data.get("function_call"):
                    if data["function_call"] == "auto":
                        data["tool_choice"] = "auto"
                    elif data["function_call"] == "none":
                        data["tool_choice"] = "none"
                    else:
                        func_data = data["function_call"]
                        if isinstance(func_data, str) and func_data != "auto" and func_data != "none":
                            data["tool_choice"] = {"type": "function", "function": {"name": func_data}}
                        elif isinstance(func_data, dict) and "name" in func_data:
                            data["tool_choice"] = {"type": "function", "function": {"name": func_data["name"]}}
                        else:
                            data["tool_choice"] = "auto"
        
        # 使用配置中的模型
        if not data.get("model"):
            data["model"] = self.config.model
        
        # 应用默认参数
        if data.get("max_tokens") is None and hasattr(self.config, "max_tokens"):
            data["max_tokens"] = self.config.max_tokens
        if data.get("temperature") is None:
            data["temperature"] = self.config.temperature
            
        # 添加extra_body配置（如果存在）
        if hasattr(self.config, "extra_body") and self.config.extra_body:
            if "extra_body" in data:
                data["extra_body"].update(self.config.extra_body)
            else:
                data["extra_body"] = self.config.extra_body
            logger.debug(f"应用extra_body配置: {data['extra_body']}")
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # 改进的API Key处理逻辑
        api_key = data.get('api_key', getattr(self.config, 'api_key', 'dummy-key'))
        
        if self.service_type == "ollama":
            headers["Authorization"] = "Bearer ollama"
        elif self.service_type == "anthropic":
            headers["x-api-key"] = api_key
            # 针对Anthropic的特殊处理
            if hasattr(self.config, "extra_body") and self.config.extra_body and "enable_thinking" in self.config.extra_body:
                headers["anthropic-beta"] = "thinking-in-llms-v0"
        elif self.service_type == "qwen":
            headers["Authorization"] = f"Bearer {api_key}"
            # 针对Qwen的特殊处理
            if hasattr(self.config, "extra_body") and self.config.extra_body and "enable_thinking" in self.config.extra_body:
                headers["X-DashScope-Beta"] = "thinking-in-llms-v1"
        elif self.service_type == "azure":
            headers["api-key"] = api_key
        else:
            # 其他服务默认使用Bearer认证
            headers["Authorization"] = f"Bearer {api_key}"
        
        timeout = aiohttp.ClientTimeout(
            total=getattr(self.config, 'timeout', 300) * 2,  # 总超时为配置值的 2 倍
            connect=30,  # 连接超时 30 秒
            sock_read=getattr(self.config, 'timeout', 300)  # 读取超时使用配置值
        )
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # 始终使用流式响应
                # 确保stream参数始终为True
                data["stream"] = True
                
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=data,
                    headers=headers
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.content:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # 解析 SSE 格式
                        if line.startswith(b"data: "):
                            content = line[6:]
                            if content == b"[DONE]":
                                return  # 这个return是允许的，因为它没有返回值
                            
                            try:
                                chunk = json.loads(content)
                                # 打印原始响应结果
                                # todo 原始打印单独记录
                                # logger.info(f"模型原始响应: {json.dumps(chunk, ensure_ascii=False)}")
                                yield chunk
                            except json.JSONDecodeError as e:
                                logger.error(f"解析流式响应失败: {e}")
                                continue
        
        except aiohttp.ClientError as e:
            logger.error(f"API 请求失败: {e}")
            raise
        except Exception as e:
            logger.error(f"未知错误: {e}")
            raise
    
    async def health_check(self) -> bool:
        """检查 API 服务是否可用"""
        try:
            # 对于不同的服务用不同的方式开展健康检查
            if self.service_type in ["ollama", "lmstudio", "local"]:
                # 本地服务使用HTTP请求
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                
                # 根据服务类型设置不同的认证头
                if self.service_type == "ollama":
                    headers["Authorization"] = "Bearer ollama"
                elif hasattr(self.config, 'api_key') and self.config.api_key:
                    headers["Authorization"] = f"Bearer {self.config.api_key}"
                
                timeout = aiohttp.ClientTimeout(total=10)  # 设置相对短的超时
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    try:
                        async with session.get(f"{self.base_url}/models", headers=headers) as response:
                            return response.status == 200
                    except:
                        logger.error(f"无法连接服务: {self.base_url}")
                        return False
            else:
                # 使用OpenAI SDK进行检查
                try:
                    models = await self.client.models.list()
                    return len(models.data) > 0
                except Exception as e:
                    logger.error(f"SDK健康检查失败: {e}")
                    
                    # 尝试使用HTTP检查
                    try:
                        timeout = aiohttp.ClientTimeout(total=10)
                        
                        # 根据服务类型设置不同的认证头
                        if self.service_type == "azure":
                            headers = {"api-key": self.config.api_key or 'dummy-key'}
                        elif self.service_type == "anthropic":
                            headers = {"x-api-key": self.config.api_key or 'dummy-key'}
                        else:
                            headers = {"Authorization": f"Bearer {self.config.api_key or 'dummy-key'}"}
                        
                        async with aiohttp.ClientSession(timeout=timeout) as session:
                            async with session.get(f"{self.base_url}/models", headers=headers) as response:
                                return response.status == 200
                    except:
                        return False
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False
    
    async def get_models(self) -> Optional[ModelListResponse]:
        """获取可用模型列表"""
        try:
            # 针对不同服务类型使用不同的获取方式
            if self.service_type in ["ollama", "lmstudio", "local"]:
                # 使用HTTP请求获取模型列表
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                
                # 根据服务类型设置不同的认证头
                if self.service_type == "ollama":
                    headers["Authorization"] = "Bearer ollama"
                elif hasattr(self.config, 'api_key') and self.config.api_key:
                    headers["Authorization"] = f"Bearer {self.config.api_key}"
                
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(f"{self.base_url}/models", headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            # 结果可能是列表或者带 data 字段的对象
                            if isinstance(data, list):
                                models = data
                            elif isinstance(data, dict) and "data" in data:
                                models = data["data"]
                            # Ollama 特殊情况
                            elif isinstance(data, dict) and "models" in data:
                                models = data["models"]
                            else:
                                models = []
                            
                            # 转换为标准格式
                            model_objects = []
                            for model in models:
                                if isinstance(model, dict):
                                    # 适配不同的模型名称字段
                                    model_id = model.get("id") or model.get("name") or model.get("model")
                                    if model_id:
                                        model_objects.append(
                                            ModelObject(
                                                id=model_id,
                                                object="model",
                                                created=model.get("created", int(time.time())),
                                                owned_by=model.get("owned_by", "local")
                                            )
                                        )
                                elif isinstance(model, str):
                                    # 如果只是字符串，直接使用
                                    model_objects.append(
                                        ModelObject(
                                            id=model,
                                            object="model",
                                            created=int(time.time()),
                                            owned_by="local"
                                        )
                                    )
                            
                            if model_objects:
                                return ModelListResponse(
                                    object="list",
                                    data=model_objects
                                )
            else:
                # 使用OpenAI SDK获取模型列表
                try:
                    models_response = await self.client.models.list()
                    
                    model_objects = []
                    for model in models_response.data:
                        model_objects.append(
                            ModelObject(
                                id=model.id,
                                object="model",
                                created=int(model.created or time.time()),
                                owned_by=model.owned_by or "unknown"
                            )
                        )
                    
                    if model_objects:
                        return ModelListResponse(
                            object="list",
                            data=model_objects
                        )
                except Exception as e:
                    logger.error(f"SDK获取模型列表失败: {e}")
                    
                    # 尝试使用HTTP请求
                    try:
                        # 根据服务类型设置不同的认证头
                        if self.service_type == "azure":
                            headers = {"api-key": self.config.api_key or 'dummy-key'}
                        elif self.service_type == "anthropic":
                            headers = {"x-api-key": self.config.api_key or 'dummy-key'}
                        else:
                            headers = {"Authorization": f"Bearer {self.config.api_key or 'dummy-key'}"}
                            
                        timeout = aiohttp.ClientTimeout(total=10)
                        async with aiohttp.ClientSession(timeout=timeout) as session:
                            async with session.get(f"{self.base_url}/models", headers=headers) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    models = data.get("data", [])
                                    
                                    model_objects = []
                                    for model in models:
                                        if isinstance(model, dict):
                                            model_objects.append(
                                                ModelObject(
                                                    id=model.get("id", "unknown"),
                                                    object="model",
                                                    created=model.get("created", int(time.time())),
                                                    owned_by=model.get("owned_by", "unknown")
                                                )
                                            )
                                    
                                    if model_objects:
                                        return ModelListResponse(
                                            object="list",
                                            data=model_objects
                                        )
                    except:
                        pass
                        
            return None
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            return None
