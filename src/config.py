import json
import os
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, ValidationError


class ServerConfig(BaseModel):
    """服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    api_prefix: str = "/v1"
    cors: Dict[str, Any] = {"enabled": True, "origins": ["*"]}


class LLMServiceConfig(BaseModel):
    """LLM 服务配置"""
    base_url: str = "https://api.openai.com/v1"  # 默认使用OpenAI API
    model: str = "gpt-4o"                       # 默认模型
    timeout: int = 60                          # 超时时间（秒）
    max_tokens: int = 4096                     # 最大生成长度
    temperature: float = 0.1                   # 温度参数
    api_key: Optional[str] = None              # API密钥
    max_retries: int = 2                       # 最大重试次数
    service_type: Optional[str] = None         # 服务类型（openai、ollama、lmstudio、anthropic等）
    
    class Config:
        extra = "allow"  # 允许额外的字段，以兼容不同服务的特殊参数


class MCPServerConfig(BaseModel):
    """MCP 服务器配置"""
    id: str
    name: str
    command: str
    args: List[str]
    enabled: bool = True
    env: Optional[Dict[str, str]] = None  # 支持环境变量
    timeout: Optional[int] = 30  # 超时设置（秒）
    init_timeout: Optional[int] = 10  # 初始化超时（秒）


class SessionConfig(BaseModel):
    """会话配置"""
    timeout: int = 3600
    max_active: int = 100


class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = "info"
    # file: str = "mcp-framework.log"


class Config(BaseModel):
    """主配置类"""
    server: ServerConfig
    llm_service: LLMServiceConfig
    mcp_servers: List[MCPServerConfig]
    sessions: SessionConfig
    logging: LoggingConfig

    @classmethod
    def load_from_file(cls, config_path: str = "config.json") -> "Config":
        """从文件加载配置"""
        if not os.path.exists(config_path):
            # 创建默认配置
            default_config = cls(
                server=ServerConfig(),
                llm_service=LLMServiceConfig(),
                mcp_servers=[],
                sessions=SessionConfig(),
                logging=LoggingConfig()
            )

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config.dict(), f, indent=2, ensure_ascii=False)

            return default_config

        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        try:
            return cls(**config_data)
        except ValidationError as e:
            raise ValueError(f"配置文件格式错误: {e}")

    def save_to_file(self, config_path: str = "config.json"):
        """保存配置到文件"""
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.dict(), f, indent=2, ensure_ascii=False)


# 全局配置实例
_config: Config = None


def get_config() -> Config:
    """获取全局配置实例"""
    global _config
    if _config is None:
        _config = Config.load_from_file()
    return _config


def reload_config():
    """重新加载配置"""
    global _config
    _config = Config.load_from_file()
