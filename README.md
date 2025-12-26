# LLM Client Library

一个简洁通用的LLM客户端库，支持所有OpenAI兼容的API接口。

## 特性

- 🚀 基于OpenAI官方SDK，稳定可靠
- 🔌 支持所有OpenAI兼容的API（OpenAI、Azure、国内各大模型等）
- 💬 支持流式和非流式对话
- 🤖 内置Agent系统，可创建具有特定角色的AI助手
- 📝 自动管理对话历史
- ⚙️ 灵活的配置系统

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 配置API

编辑 `config.yaml` 文件：

```yaml
api:
  api_key: "your-api-key"
  base_url: "https://api.openai.com/v1"  # 或其他兼容的API地址
  model: "gpt-3.5-turbo"
  temperature: 0.7
  max_tokens: 2000
```

### 2. 基础对话

```python
from llm_client import LLMClient
from conversation import Conversation

# 创建客户端
client = LLMClient(
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",
    model="gpt-3.5-turbo"
)

# 创建对话
conv = Conversation(client)

# 发送消息
response = conv.send("Hello!")
print(response)
```

### 3. 流式对话

```python
# 流式输出
for chunk in conv.stream_send("Tell me a story"):
    print(chunk, end="", flush=True)
```

### 4. 创建AI Agent

```python
from agent import Agent

# 创建具有特定角色的Agent
teacher = Agent(
    client=client,
    name="Teacher",
    role="You are a patient teacher.",
    personality="Friendly and encouraging"
)

response = teacher.respond("Explain quantum physics")
print(response)
```

## 核心组件

### LLMClient

通用LLM客户端，支持所有OpenAI兼容的API。

**参数：**
- `api_key`: API密钥
- `base_url`: API基础URL
- `model`: 模型名称
- `temperature`: 温度参数（0-1）
- `max_tokens`: 最大token数
- `**kwargs`: 其他额外参数

**方法：**
- `chat(messages, **kwargs)`: 发送对话请求，返回完整响应
- `stream_chat(messages, **kwargs)`: 流式发送对话请求

### Conversation

对话管理器，自动维护对话历史。

**参数：**
- `client`: LLMClient实例
- `system_prompt`: 系统提示词（可选）

**方法：**
- `send(message, **kwargs)`: 发送消息并获取响应
- `stream_send(message, **kwargs)`: 流式发送消息
- `clear()`: 清空对话历史
- `get_history()`: 获取对话历史
- `set_system_prompt(prompt)`: 设置系统提示词

### Agent

AI代理，具有特定角色和个性。

**参数：**
- `client`: LLMClient实例
- `name`: Agent名称
- `role`: 角色描述
- `personality`: 个性描述

**方法：**
- `respond(message, **kwargs)`: 生成响应
- `stream_respond(message, **kwargs)`: 流式生成响应
- `reset()`: 重置对话历史
- `get_history()`: 获取对话历史

## 使用示例

### 示例1：简单对话

```python
from llm_client import LLMClient
from conversation import Conversation

client = LLMClient(
    api_key="sk-xxx",
    base_url="https://api.openai.com/v1",
    model="gpt-3.5-turbo"
)

conv = Conversation(client, system_prompt="You are a helpful assistant.")
response = conv.send("What's the weather like?")
print(response)
```

### 示例2：多轮对话

```python
conv = Conversation(client)

conv.send("My name is Alice")
conv.send("What's my name?")  # AI会记住你的名字
```

### 示例3：创建专业Agent

```python
from agent import Agent

# 创建代码助手
coder = Agent(
    client=client,
    name="Coder",
    role="You are an expert programmer.",
    personality="Concise and technical"
)

code = coder.respond("Write a Python function to calculate fibonacci")
print(code)
```

### 示例4：使用不同的API提供商

```python
# 使用Azure OpenAI
client = LLMClient(
    api_key="your-azure-key",
    base_url="https://your-resource.openai.azure.com/openai/deployments/your-deployment",
    model="gpt-35-turbo"
)

# 使用国内API（如智谱、通义等）
client = LLMClient(
    api_key="your-key",
    base_url="https://api.provider.com/v1",
    model="model-name"
)
```

### 示例5：自定义参数

```python
# 创建时设置默认参数
client = LLMClient(
    api_key="sk-xxx",
    base_url="https://api.openai.com/v1",
    model="gpt-4",
    temperature=0.9,
    max_tokens=4000
)

# 调用时覆盖参数
response = conv.send("Tell me a story", temperature=1.0, max_tokens=1000)
```

## 运行示例

```bash
# 运行交互式聊天
python chat.py

# 运行简单对话示例
python examples/simple_chat.py

# 运行Agent示例
python examples/agent_example.py
```

## 兼容的API提供商

本库支持所有OpenAI兼容的API，包括但不限于：

- OpenAI
- Azure OpenAI
- 智谱AI (GLM)
- 通义千问 (Qwen)
- 文心一言 (ERNIE)
- 讯飞星火 (Spark)
- Moonshot AI
- DeepSeek
- 其他提供OpenAI兼容接口的服务

## 高级用法

### 自定义系统提示词

```python
conv = Conversation(
    client,
    system_prompt="You are a professional translator. Translate everything to Chinese."
)
```

### 管理对话历史

```python
# 获取历史
history = conv.get_history()

# 清空历史
conv.clear()

# 修改系统提示词
conv.set_system_prompt("New system prompt")
```

### 流式输出控制

```python
full_response = ""
for chunk in conv.stream_send("Write a poem"):
    full_response += chunk
    print(chunk, end="", flush=True)
print(f"\n\nFull response length: {len(full_response)}")
```

## 项目结构

```
LLM/
├── llm_client.py      # 核心LLM客户端
├── conversation.py    # 对话管理
├── agent.py          # Agent系统
├── chat.py           # 交互式聊天界面
├── config.yaml       # 配置文件
├── requirements.txt  # 依赖
├── README.md         # 使用手册
└── examples/         # 示例代码
    ├── simple_chat.py
    └── agent_example.py
```

## 注意事项

1. 请妥善保管API密钥，不要提交到版本控制系统
2. 不同的API提供商可能有不同的模型名称和参数要求
3. 注意API调用的费用和速率限制
4. 流式输出时需要正确处理异常

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！
