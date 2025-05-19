#!/usr/bin/env python3
"""
MCP Framework 启动脚本

"""

import os
import sys

# 添加 src 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import main

if __name__ == "__main__":
    main()
