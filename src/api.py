import asyncio
import json
import logging
import time
from typing import Optional

import aiohttp
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from chat_handler import ChatHandler
from config import get_config
from llm_service_client import LLMServiceClient
from mcp_manager import MCPManager
from models import (
    ChatCompletionRequest,
    ErrorResponse,
    ModelListResponse,
    ModelObject
)
from session_manager import SessionManager

logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    config = get_config()
    
    # 创建 FastAPI 实例
    app = FastAPI(
        title="MCP Framework API",
        description="OpenAI 兼容的 API 接口，集成 MCP 服务器和 LLM 服务",
        version="1.0.0"
    )
    
    # 配置 CORS
    if config.server.cors.get("enabled", True):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.server.cors.get("origins", ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # 初始化组件
    llm_service_client = LLMServiceClient()
    mcp_manager = MCPManager()
    session_manager = SessionManager(
        timeout=config.sessions.timeout,
        max_active=config.sessions.max_active
    )
    chat_handler = ChatHandler(llm_service_client, mcp_manager, session_manager)
    
    # 添加到 app 状态
    app.state.llm_service_client = llm_service_client
    app.state.mcp_manager = mcp_manager
    app.state.session_manager = session_manager
    app.state.chat_handler = chat_handler
    
    # 启动时初始化 MCP 服务器
    @app.on_event("startup")
    async def startup_event():
        """启动时初始化"""
        logger.info("正在启动 MCP Framework...")
        
        # 检查 LLM 服务连接
        if not await llm_service_client.health_check():
            logger.warning(f"LLM 服务 ({config.llm_service.base_url}) 未响应，请确保服务已启动")
        else:
            logger.info(f"LLM 服务连接成功: {config.llm_service.base_url}")
        
        # 启动配置的 MCP 服务器
        for server_config in config.mcp_servers:
            if server_config.enabled:
                try:
                    success = await mcp_manager.start_server(server_config)
                    if not success:
                        logger.error(f"无法启动 MCP 服务器: {server_config.id}")
                except Exception as e:
                    logger.error(f"启动 MCP 服务器 {server_config.id} 时出错: {e}", exc_info=True)
        
        # 显示可用工具
        tools = mcp_manager.get_all_tools()
        logger.info(f"已加载 {len(tools)} 个 MCP 工具")
        if tools:
            # 适配新旧两种工具格式
            tool_names = []
            for tool in tools[:5]:
                if "type" in tool and tool["type"] == "function" and "function" in tool:
                    # 新格式工具
                    tool_names.append(tool["function"].get("name", "unknown"))
                else:
                    # 旧格式工具或其他格式
                    tool_names.append(tool.get("name", "unknown"))
            logger.debug(f"可用工具: {tool_names}...")
        
        logger.info("MCP Framework 启动完成")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """关闭时清理"""
        logger.info("正在关闭 MCP Framework...")
        await mcp_manager.stop_all_servers()
        logger.info("MCP Framework 已关闭")
    
    # API 路由
    @app.get("/health")
    async def health_check():
        """健康检查"""
        llm_service_healthy = await llm_service_client.health_check()
        active_servers = len(mcp_manager.servers)
        active_sessions = session_manager.get_active_sessions_count()
        
        return {
            "status": "healthy",
            "components": {
                "llm_service": "connected" if llm_service_healthy else "disconnected",
                "llm_service_url": config.llm_service.base_url,
                "mcp_servers": active_servers,
                "active_sessions": active_sessions
            }
        }
    
    @app.get(f"{config.server.api_prefix}/models")
    async def list_models():
        """获取可用模型列表"""
        try:
            # 先尝试从 LLM 服务获取模型列表
            models = await llm_service_client.get_models()
            if models:
                return models
            else:
                # 如果获取失败，返回配置的默认模型
                return ModelListResponse(
                    object="list",
                    data=[
                        ModelObject(
                            id=config.llm_service.model,
                            object="model",
                            created=int(time.time()),
                            owned_by="local"
                        )
                    ]
                )
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            # 出错时返回配置的默认模型
            return ModelListResponse(
                object="list",
                data=[
                    ModelObject(
                        id=config.llm_service.model,
                        object="model",
                        created=int(time.time()),
                        owned_by="local"
                    )
                ]
            )
    
    @app.post(f"{config.server.api_prefix}/chat/completions")
    async def chat_completions(
        request: ChatCompletionRequest,
        authorization: Optional[str] = Header(None),
        x_session_id: Optional[str] = Header(None)
    ):
        """主要的聊天完成接口（OpenAI 兼容）"""
        try:
            # 处理会话 ID
            session_id = x_session_id
            
            # 调用处理器 - 始终使用流式响应
            response_iterator = chat_handler.handle_chat_completion(request, session_id)
            
            # 流式响应
            async def generate():
                async for chunk in response_iterator:
                    if isinstance(chunk, dict):
                        yield f"data: {json.dumps(chunk)}\n\n"
                    else:
                        yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        
        except asyncio.TimeoutError:
            logger.error("Request timeout")
            return JSONResponse(
                status_code=504,
                content=ErrorResponse(
                    error={
                        "message": "Request timeout. Please try with smaller max_tokens or simpler prompts.",
                        "type": "timeout_error",
                        "param": None,
                        "code": "timeout"
                    }
                ).dict()
            )
        except Exception as e:
            logger.error(f"处理请求失败: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error={
                        "message": f"处理请求失败: {str(e)}",
                        "type": "server_error",
                        "param": None,
                        "code": "internal_error"
                    }
                ).dict()
            )
        except aiohttp.ClientError as e:
            logger.error(f"Client error: {e}", exc_info=True)
            return JSONResponse(
                status_code=502,
                content=ErrorResponse(
                    error={
                        "message": f"Service connection error: {str(e)}",
                        "type": "service_error",
                        "param": None,
                        "code": "connection_error"
                    }
                ).dict()
            )
    
    # 兼容性路由（没有版本前缀）
    @app.post("/chat/completions")
    async def chat_completions_no_prefix(
        request: ChatCompletionRequest,
        authorization: Optional[str] = Header(None),
        x_session_id: Optional[str] = Header(None)
    ):
        """兼容性路由（无版本前缀）"""
        return await chat_completions(request, authorization, x_session_id)
    
    @app.get("/models")
    async def list_models_no_prefix():
        """兼容性路由（无版本前缀）"""
        return await list_models()
    
    return app

# 创建应用实例
app = create_app()
