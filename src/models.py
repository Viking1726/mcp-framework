import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """OpenAI 标准消息格式"""
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class FunctionParameter(BaseModel):
    """函数参数定义"""
    type: str
    properties: Dict[str, Any]
    required: List[str] = Field(default_factory=list)

class FunctionDefinition(BaseModel):
    """OpenAI 函数定义格式"""
    name: str
    description: str
    parameters: Dict[str, Any]  # 直接使用 Dict，不是 FunctionParameter

class ChatCompletionRequest(BaseModel):
    """OpenAI Chat Completion 请求格式"""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[List[str]] = None
    stream: Optional[bool] = True
    # 兼容旧版OpenAI的functions格式
    functions: Optional[List[FunctionDefinition]] = None
    function_call: Optional[str] = None
    # 新版OpenAI SDK的tools格式
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    user: Optional[str] = None

class ChatCompletionResponseMessage(BaseModel):
    """响应消息格式"""
    role: str
    content: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None

class ChatCompletionChoice(BaseModel):
    """响应选项格式"""
    index: int
    message: ChatCompletionResponseMessage
    finish_reason: Optional[str] = None

class Usage(BaseModel):
    """Token 使用统计"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    """OpenAI Chat Completion 响应格式"""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage

class ChatCompletionChunk(BaseModel):
    """流式响应块格式"""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]

class ErrorResponse(BaseModel):
    """错误响应格式（OpenAI 标准）"""
    error: Dict[str, Any]

class ModelObject(BaseModel):
    """模型对象格式"""
    id: str
    object: str = "model"
    created: int
    owned_by: str

class ModelListResponse(BaseModel):
    """模型列表响应格式"""
    object: str = "list"
    data: List[ModelObject]

class MCPTool(BaseModel):
    """MCP 工具定义"""
    name: str
    description: str
    inputSchema: Dict[str, Any]

class MCPRequest(BaseModel):
    """MCP 请求格式"""
    method: str
    params: Dict[str, Any]

class MCPResponse(BaseModel):
    """MCP 响应格式"""
    result: Optional[Any] = None
    error: Optional[Dict[str, str]] = None
