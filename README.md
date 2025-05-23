![截屏2025-05-19 下午4.28.07.png](%E6%88%AA%E5%B1%8F2025-05-19%20%E4%B8%8B%E5%8D%884.28.07.png)

# MCP框架架构文档

## 1. 概述

MCP框架（Model Context Protocol Framework）是一个用于连接大语言模型（LLM）服务和工具的中间件系统。它提供了一个与OpenAI API兼容的接口，允许客户端应用通过标准API与不同的LLM服务进行交互，同时集成多种工具服务（MCP服务器）以增强AI助手的能力。

核心特性：
- 兼容OpenAI的聊天完成API(OpenAI Python SDK)
- 支持多种LLM服务提供商
- 插件化工具集成架构(ModelContextProtocol Python SDK)
- 会话管理
- 交互式工具调用

## 2. 系统架构

MCP框架采用模块化的架构设计，主要由以下几个核心组件构成：

### 系统架构图

```
客户端应用
    │
    ▼
┌───────────────────────────────────────────────┐
│                                               │
│               MCP框架 (FastAPI)               │
│                                               │
├───────────┬─────────────┬───────────┬─────────┤
│           │             │           │         │
│  API层    │ 聊天处理器  │ 会话管理器 │ 配置管理 │
│           │             │           │         │
├───────────┴──────┬──────┴───────────┴─────────┤
│                  │                            │
│  LLM服务客户端    │        MCP管理器           │
│                  │                            │
└──────┬───────────┴────────────┬───────────────┘
       │                        │
       ▼                        ▼
┌─────────────────┐    ┌─────────────────────────┐
│                 │    │                         │
│  LLM服务        │    │  MCP工具服务            │
│  (OpenAI等)     │    │  (文件系统、时间等)      │
│                 │    │                         │
└─────────────────┘    └─────────────────────────┘
```

架构说明：
- 客户端应用通过HTTP API与MCP框架交互
- MCP框架处理请求并协调LLM服务和工具服务
- LLM服务客户端负责与不同的LLM提供商通信
- MCP管理器负责启动、管理和调用各种工具服务
- 会话管理器维护客户端会话状态
- 配置管理负责系统设置和参数

### 2.1 核心组件

#### LLM服务客户端 (LLMServiceClient)
负责与各种LLM服务（如OpenAI、Anthropic、Ollama、LM Studio等）进行通信。具有如下特性：
- 自动检测服务类型
- 统一的API调用接口
- 流式响应处理
- 健康检查和模型列表查询

#### MCP管理器 (MCPManager)
管理多个MCP工具服务器的连接和工具调用。主要功能：
- 启动和管理多个工具服务器
- 解析工具名称并路由工具调用
- 处理工具执行结果
- 提供系统中所有可用工具的列表

#### 会话管理器 (SessionManager)
维护客户端会话状态。功能包括：
- 创建和管理会话
- 存储会话消息历史
- 会话超时处理和自动清理
- 会话数量限制

#### 聊天处理器 (ChatHandler)
处理客户端的聊天请求，协调LLM服务和工具调用。主要功能：
- 处理聊天完成请求
- 管理工具交互
- 处理流式响应
- 构建系统提示和上下文

#### Web API (FastAPI应用)
提供与OpenAI兼容的REST API接口，用于客户端与系统交互。主要端点：
- `/chat/completions` - 聊天完成主端点
- `/models` - 获取可用模型列表
- `/health` - 系统健康检查

### 2.2 数据模型

系统使用Pydantic模型定义各种数据结构，主要包括：
- `ChatMessage` - 聊天消息格式
- `ChatCompletionRequest` - 客户端请求
- `ChatCompletionResponse` - API响应
- `ModelObject` - 模型信息
- `ModelListResponse` - 模型列表响应

## 3. 工作流程

### 3.1 启动流程

1. 程序执行入口为`run.py`，调用`main.py`中的`main()`函数
2. 设置日志系统
3. 加载配置文件(`config.json`)
4. 创建FastAPI应用
5. 初始化主要组件：
   - LLM服务客户端
   - MCP管理器
   - 会话管理器
   - 聊天处理器
6. 启动配置的MCP服务器
7. 启动Web服务器

### 3.2 请求处理流程

1. 客户端发送聊天完成请求到`/chat/completions`端点
2. API层接收请求并创建会话（如果不存在）
3. 聊天处理器增强请求（添加工具信息和系统提示）
4. 请求发送到LLM服务
5. 处理LLM响应并检查工具调用
6. 如有工具调用，执行工具并收集结果
7. 将工具结果添加到上下文
8. 继续与LLM交互直到完成任务
9. 将最终结果以流式方式返回给客户端

## 4. MCP工具服务

MCP框架支持多种工具服务器，每个服务器提供不同类型的工具。服务器通过stdio通信协议与框架交互。

当前配置的工具服务包括：
- 时间服务 (`time`) - 提供时间相关功能
- 文件系统服务 (`filesystem`) - 文件和目录操作
- 浏览器自动化服务 (`firecrawl`) - 网页抓取和自动化
- SQLite数据库服务 (`sqlite`) - 数据库操作
- 顺序思维服务 (`sequential-thinking`) - 增强模型思考能力

工具服务通过MCP客户端库(`mcp`)与框架通信，提供初始化、工具列表和执行功能。

## 5. 配置管理

系统配置使用`config.json`文件，主要包括：

- 服务器配置 - 主机、端口、CORS设置
- LLM服务配置 - 服务URL、默认模型、超时等
- MCP服务器配置 - 命令、参数、环境变量等
- 会话配置 - 超时、最大会话数
- 日志配置 - 日志级别和输出

配置由`config.py`模块加载和管理，提供全局配置访问。

## 6. 特殊功能

### 6.1 交互式工具调用

系统支持模型与工具之间的多轮交互，由`tool_interactive.py`实现：
- 存储工具执行结果
- 解析参数中的结果引用
- 提供上下文信息给模型

这使模型能够：
1. 调用工具获取信息
2. 查看执行结果
3. 基于结果决定下一步操作
4. 在后续工具调用中引用之前的结果

### 6.2 LLM服务适配

系统能够自动适配多种LLM服务：
- OpenAI API
- Azure OpenAI
- Anthropic
- Ollama
- LM Studio
- 通义千问(Qwen)

每种服务都有特定的认证方式和参数格式，系统自动处理这些差异。

## 7. 扩展性

MCP框架设计为易于扩展：

1. **添加新工具服务**：在配置中添加新的MCP服务器定义
2. **支持新的LLM服务**：扩展LLMServiceClient以添加新服务类型
3. **增加API功能**：在FastAPI应用中添加新端点
4. **自定义系统提示**：修改ChatHandler中的提示模板

## 8. 部署

系统支持多种部署方式：

- **本地运行**：直接执行`run.py`
- **Docker容器**：使用提供的Dockerfile和docker-compose.yml
- **云服务**：可部署在任何支持Python的云平台

系统依赖项在`requirements.txt`中定义。

## 9. 安全考虑

- 所有对LLM服务的请求都通过框架代理
- MCP工具服务以子进程形式运行，受限于框架的权限
- 会话有超时机制，防止资源耗尽
- 支持CORS设置以控制访问来源

## 10. 总结

MCP框架提供了一个灵活、可扩展的架构，用于集成大语言模型和工具服务。它使开发者能够创建功能强大的AI助手应用，同时保持系统的模块化和可维护性。

核心优势：
- 兼容OpenAI API的标准接口
- 灵活的工具集成机制
- 支持多种LLM服务提供商
- 交互式工具调用能力
- 模块化和可扩展的架构
