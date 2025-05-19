import logging

import uvicorn

from api import app
from config import get_config


# 配置日志
def setup_logging():
    """设置日志配置"""
    config = get_config()
    
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # logging.FileHandler(config.logging.file),
            logging.StreamHandler()
        ]
    )

def main():
    """主程序入口"""
    try:
        # 设置日志
        setup_logging()
        
        # 获取配置
        config = get_config()
        
        logger = logging.getLogger(__name__)
        logger.info(f"MCP Framework 启动")
        logger.info(f"服务器地址: {config.server.host}:{config.server.port}")
        logger.info(f"LLM 服务 URL: {config.llm_service.base_url}")
        logger.info(f"默认模型: {config.llm_service.model}")
        logger.info(f"MCP 服务器数量: {len(config.mcp_servers)}")
        
        # 启动服务器
        uvicorn.run(
            app,
            host=config.server.host,
            port=config.server.port,
            log_level=config.logging.level.lower()
        )
    except Exception as e:
        print(f"\n启动失败: {e}")
        print("\n请检查:")
        print("1. 是否安装了所有依赖: pip install -r requirements.txt")
        print("2. 配置文件是否正确: config.json")
        print("3. 端口是否被占用")
        print("\n详细错误:")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
