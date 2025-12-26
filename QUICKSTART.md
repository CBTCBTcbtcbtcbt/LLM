# 快速入门指南

## 5分钟上手

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

编辑 `config.yaml`：

```yaml
api:
  api_key: "your-api-key-here"
  base_url: "https://api.openai.com/v1"
  model: "gpt-3.5-turbo"
  temperature: 0.7
  max_tokens: 2000
```

### 3. 运行聊天程序

```bash
python chat.py
```

## 代码示例

### 最简单的使用方式

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
response = conv.send("你好！")
print(response)
```

### 使用配置文件

```python
from llm_client import LLMClient
from conversation import Conversation
import yaml

# 加载配置
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

api = config['api']

# 创建客户端
client = LLMClient(
    api_key=api['api_key'],
    base_url=api['base_url'],
    model=api['model']
)

# 开始对话
conv = Conversation(client)
response = conv.send("介绍一下Python")
print(response)
```

### 流式输出

```python
# 流式输出，实时显示
for chunk in conv.stream_send("写一首诗"):
    print(chunk, end="", flush=True)
```

### 创建AI助手

```python
from agent import Agent

# 创建翻译助手
translator = Agent(
    client=client,
    name="Translator",
    role="You are a professional translator.",
    personality="Accurate and natural"
)

# 使用助手
result = translator.respond("Translate to English: 你好世界")
print(result)
```

## 常见API提供商配置

### OpenAI官方

```yaml
api:
  api_key: "sk-..."
  base_url: "https://api.openai.com/v1"
  model: "gpt-3.5-turbo"
```

### Azure OpenAI

```yaml
api:
  api_key: "your-azure-key"
  base_url: "https://your-resource.openai.azure.com/openai/deployments/your-deployment"
  model: "gpt-35-turbo"
```

### 国内API（通用格式）

```yaml
api:
  api_key: "your-key"
  base_url: "https://api.provider.com/v1"
  model: "model-name"
```

## 下一步

- 查看 [README.md](README.md) 了解完整文档
- 运行 `examples/` 目录下的示例代码
- 根据需求自定义Agent角色

## 常见问题

**Q: 如何切换不同的模型？**

A: 修改 `config.yaml` 中的 `model` 字段即可。

**Q: 支持哪些API？**

A: 支持所有OpenAI兼容的API接口。

**Q: 如何调整回复的创造性？**

A: 修改 `temperature` 参数（0-1），值越大越有创造性。

**Q: 如何限制回复长度？**

A: 修改 `max_tokens` 参数。
