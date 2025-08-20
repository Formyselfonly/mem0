# Mem0 代码架构说明

## 📁 项目结构

```
mem0/
├── client/           # API 客户端层
│   ├── main.py      # 同步/异步客户端实现
│   ├── project.py   # 项目管理功能
│   └── utils.py     # 客户端工具函数
├── memory/           # 核心记忆管理模块
│   ├── main.py      # 记忆系统核心逻辑 (1866行)
│   ├── base.py      # 记忆基类
│   ├── storage.py   # 本地存储管理
│   ├── utils.py     # 记忆处理工具
│   ├── telemetry.py # 遥测数据收集
│   ├── setup.py     # 初始化配置
│   ├── graph_memory.py    # 图记忆实现
│   ├── kuzu_memory.py     # Kuzu 图数据库集成
│   └── memgraph_memory.py # Memgraph 图数据库集成
├── configs/          # 配置管理
│   ├── base.py      # 基础配置类
│   ├── enums.py     # 枚举定义
│   ├── prompts.py   # 提示词模板
│   ├── llms/        # LLM 配置
│   ├── embeddings/  # 嵌入模型配置
│   └── vector_stores/ # 向量数据库配置
├── llms/            # LLM 集成层
│   ├── base.py      # LLM 基类
│   ├── configs.py   # LLM 配置
│   ├── openai.py    # OpenAI 集成
│   ├── anthropic.py # Anthropic 集成
│   ├── azure_openai.py # Azure OpenAI 集成
│   ├── aws_bedrock.py # AWS Bedrock 集成
│   ├── gemini.py    # Google Gemini 集成
│   ├── ollama.py    # Ollama 本地模型
│   ├── lmstudio.py  # LM Studio 集成
│   ├── groq.py      # Groq 集成
│   ├── together.py  # Together AI 集成
│   ├── deepseek.py  # DeepSeek 集成
│   ├── vllm.py      # vLLM 集成
│   ├── litellm.py   # LiteLLM 集成
│   ├── sarvam.py    # Sarvam 集成
│   ├── xai.py       # XAI 集成
│   └── langchain.py # LangChain 集成
├── vector_stores/   # 向量数据库集成
│   ├── base.py      # 向量存储基类
│   ├── configs.py   # 向量存储配置
│   ├── pinecone.py  # Pinecone 集成
│   ├── weaviate.py  # Weaviate 集成
│   ├── qdrant.py    # Qdrant 集成
│   ├── chroma.py    # Chroma 集成
│   ├── faiss.py     # FAISS 集成
│   ├── supabase.py  # Supabase 集成
│   ├── mongodb.py   # MongoDB 集成
│   ├── redis.py     # Redis 集成
│   ├── elasticsearch.py # Elasticsearch 集成
│   ├── opensearch.py # OpenSearch 集成
│   ├── pgvector.py  # PostgreSQL + pgvector
│   ├── milvus.py    # Milvus 集成
│   ├── azure_ai_search.py # Azure AI Search
│   ├── vertex_ai_vector_search.py # Google Vertex AI
│   ├── upstash_vector.py # Upstash Vector
│   ├── databricks.py # Databricks 集成
│   ├── baidu.py     # 百度向量数据库
│   └── langchain.py # LangChain 向量存储
├── embeddings/      # 嵌入模型集成
│   ├── base.py      # 嵌入模型基类
│   ├── configs.py   # 嵌入模型配置
│   ├── openai.py    # OpenAI 嵌入
│   ├── azure_openai.py # Azure OpenAI 嵌入
│   ├── cohere.py    # Cohere 嵌入
│   ├── huggingface.py # HuggingFace 嵌入
│   ├── sentence_transformers.py # Sentence Transformers
│   ├── vertex_ai.py # Google Vertex AI 嵌入
│   ├── aws_bedrock.py # AWS Bedrock 嵌入
│   ├── ollama.py    # Ollama 嵌入
│   ├── together.py  # Together AI 嵌入
│   ├── deepseek.py  # DeepSeek 嵌入
│   ├── groq.py      # Groq 嵌入
│   ├── sarvam.py    # Sarvam 嵌入
│   ├── xai.py       # XAI 嵌入
│   └── langchain.py # LangChain 嵌入
├── graphs/          # 图数据库集成
│   ├── base.py      # 图存储基类
│   ├── configs.py   # 图存储配置
│   ├── kuzu.py      # Kuzu 图数据库
│   ├── memgraph.py  # Memgraph 图数据库
│   └── neo4j.py     # Neo4j 集成
├── utils/           # 工具函数
│   ├── factory.py   # 工厂模式实现
│   └── helpers.py   # 辅助函数
└── __init__.py      # 包初始化
```

## 🧠 核心组件详解

### 1. Client 层 (`client/`)

**功能**: 提供与 Mem0 API 交互的客户端接口

**核心类**:
- `MemoryClient`: 同步客户端
- `AsyncMemoryClient`: 异步客户端

**主要方法**:
```python
# 记忆操作
client.add(messages, user_id="user123")      # 添加记忆
client.search(query, user_id="user123")      # 搜索记忆
client.get(memory_id)                        # 获取特定记忆
client.update(memory_id, text="新内容")      # 更新记忆
client.delete(memory_id)                     # 删除记忆

# 批量操作
client.batch_update(memories)                # 批量更新
client.batch_delete(memories)                # 批量删除

# 项目管理
client.project.get()                         # 获取项目信息
client.project.update(instructions="新指令") # 更新项目
```

### 2. Memory 核心模块 (`memory/`)

**功能**: 记忆系统的核心逻辑，负责记忆的创建、存储、检索和管理

**核心类**:
- `Memory`: 同步记忆管理器
- `AsyncMemory`: 异步记忆管理器

**记忆处理流程**:
```python
# 1. 消息解析和预处理
messages = parse_messages(user_input)

# 2. LLM 事实提取
facts = extract_facts_with_llm(messages)

# 3. 向量化存储
vector_store.add(facts)

# 4. 图关系构建
graph_store.add_relations(facts)

# 5. 智能检索
memories = search_relevant_memories(query)
```

**记忆类型**:
- **短期记忆**: 当前会话的临时信息
- **长期记忆**: 持久化的用户信息
- **语义记忆**: 基于含义的抽象知识
- **情节记忆**: 具体事件和经历
- **程序记忆**: 操作步骤和技能

### 3. 配置系统 (`configs/`)

**功能**: 统一管理所有组件的配置

**核心类**:
```python
class MemoryConfig:
    vector_store: VectorStoreConfig    # 向量存储配置
    llm: LlmConfig                    # LLM 配置
    embedder: EmbedderConfig          # 嵌入模型配置
    graph_store: GraphStoreConfig     # 图数据库配置
    version: str                      # API 版本
    custom_fact_extraction_prompt: str # 自定义事实提取提示词
    custom_update_memory_prompt: str  # 自定义记忆更新提示词
```

### 4. LLM 集成层 (`llms/`)

**功能**: 支持多种 LLM 提供商的统一接口

**支持的 LLM**:
- **云服务**: OpenAI, Anthropic, Azure OpenAI, AWS Bedrock, Google Gemini
- **本地部署**: Ollama, LM Studio, vLLM
- **其他服务**: Groq, Together AI, DeepSeek, Sarvam, XAI
- **框架集成**: LangChain, LiteLLM

**使用示例**:
```python
from mem0.configs.base import LlmConfig

# OpenAI 配置
llm_config = LlmConfig(
    provider="openai",
    model="gpt-4o",
    api_key="your-api-key"
)

# 本地 Ollama 配置
llm_config = LlmConfig(
    provider="ollama",
    model="llama3.2",
    base_url="http://localhost:11434"
)
```

### 5. 向量数据库集成 (`vector_stores/`)

**功能**: 支持多种向量数据库的语义搜索

**支持的数据库**:
- **云服务**: Pinecone, Weaviate, Qdrant, Supabase, MongoDB Atlas
- **开源**: Chroma, FAISS, Milvus, Elasticsearch, OpenSearch
- **企业级**: Azure AI Search, Google Vertex AI, Databricks
- **其他**: Redis, PostgreSQL (pgvector), 百度向量数据库

**使用示例**:
```python
from mem0.configs.base import VectorStoreConfig

# Pinecone 配置
vector_config = VectorStoreConfig(
    provider="pinecone",
    api_key="your-api-key",
    environment="us-west1-gcp",
    index_name="mem0-index"
)

# 本地 Chroma 配置
vector_config = VectorStoreConfig(
    provider="chroma",
    persist_directory="./chroma_db"
)
```

### 6. 嵌入模型集成 (`embeddings/`)

**功能**: 支持多种嵌入模型的文本向量化

**支持的模型**:
- **OpenAI**: text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large
- **Azure OpenAI**: 各种 Azure 嵌入模型
- **开源**: Sentence Transformers, HuggingFace 模型
- **其他**: Cohere, Google Vertex AI, AWS Bedrock, Ollama

### 7. 图数据库集成 (`graphs/`)

**功能**: 支持图数据库的关系记忆存储

**支持的数据库**:
- **Kuzu**: 高性能图数据库
- **Memgraph**: 内存图数据库
- **Neo4j**: 企业级图数据库

## 🔄 数据流

```
用户输入
    ↓
消息解析 (parse_messages)
    ↓
LLM 事实提取 (extract_facts_with_llm)
    ↓
向量化 (embeddings)
    ↓
存储到向量数据库 (vector_store.add)
    ↓
图关系提取 (extract_relations)
    ↓
存储到图数据库 (graph_store.add_relations)
    ↓
检索时结合向量和图查询
    ↓
返回相关记忆
```

## 🚀 使用示例

### 基本使用
```python
from mem0 import MemoryClient

# 初始化客户端
client = MemoryClient(api_key="your-api-key")

# 添加记忆
client.add([
    {"role": "user", "content": "我喜欢打网球"},
    {"role": "assistant", "content": "好的，我记住了你喜欢网球"}
], user_id="user123")

# 搜索记忆
memories = client.search("运动偏好", user_id="user123")
print(memories)
```

### 高级配置
```python
from mem0 import Memory
from mem0.configs.base import MemoryConfig, LlmConfig, VectorStoreConfig

# 自定义配置
config = MemoryConfig(
    llm=LlmConfig(
        provider="openai",
        model="gpt-4o",
        api_key="your-openai-key"
    ),
    vector_store=VectorStoreConfig(
        provider="pinecone",
        api_key="your-pinecone-key",
        environment="us-west1-gcp",
        index_name="mem0-index"
    ),
    custom_fact_extraction_prompt="从以下对话中提取关键事实："
)

# 初始化记忆系统
memory = Memory(config=config)

# 添加记忆
result = memory.add("用户说他喜欢在周末打网球", user_id="user123")
```

### 异步使用
```python
import asyncio
from mem0 import AsyncMemoryClient

async def main():
    async with AsyncMemoryClient(api_key="your-api-key") as client:
        # 异步添加记忆
        await client.add([
            {"role": "user", "content": "我喜欢打网球"},
            {"role": "assistant", "content": "好的，我记住了你喜欢网球"}
        ], user_id="user123")
        
        # 异步搜索记忆
        memories = await client.search("运动偏好", user_id="user123")
        print(memories)

asyncio.run(main())
```

## 🎯 核心特性

1. **多模态支持**: 支持文本、图像等多种数据类型
2. **高精度**: 比 OpenAI Memory 准确率提升 26%
3. **高性能**: 响应速度提升 91%，token 使用减少 90%
4. **可扩展性**: 支持多种 LLM 和向量数据库
5. **开发者友好**: 简洁的 API 和丰富的文档
6. **图记忆**: 支持实体关系的图数据库存储
7. **智能检索**: 结合向量搜索和图遍历的混合检索
8. **异步支持**: 支持高并发异步操作

## 🔧 开发指南

### 添加新的 LLM 支持
1. 在 `llms/` 目录下创建新的实现文件
2. 继承 `BaseLlm` 类并实现必要的方法
3. 在 `llms/__init__.py` 中注册新的 LLM
4. 更新 `configs/llms/` 中的配置

### 添加新的向量数据库支持
1. 在 `vector_stores/` 目录下创建新的实现文件
2. 继承 `BaseVectorStore` 类并实现必要的方法
3. 在 `vector_stores/__init__.py` 中注册新的向量存储
4. 更新 `configs/vector_stores/` 中的配置

### 添加新的嵌入模型支持
1. 在 `embeddings/` 目录下创建新的实现文件
2. 继承 `BaseEmbedder` 类并实现必要的方法
3. 在 `embeddings/__init__.py` 中注册新的嵌入模型
4. 更新 `configs/embeddings/` 中的配置

## 📚 相关文档

- [API 文档](https://docs.mem0.ai/api-reference)
- [快速开始](https://docs.mem0.ai/platform/quickstart)
- [配置指南](https://docs.mem0.ai/components/overview)
- [示例代码](https://github.com/mem0ai/mem0/tree/main/examples)
- [研究论文](https://mem0.ai/research)
