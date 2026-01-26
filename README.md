# LLM 框架技术文档

本文档详细介绍了该仓库根目录下核心 Python 模块的使用方法、类定义、参数说明及输入格式。

## 目录

1. [LLMClient (llm_client.py)](#1-llmclient-llm_clientpy)
2. [Conversation (conversation.py)](#2-conversation-conversationpy)
3. [Agent (agent.py)](#3-agent-agentpy)
4. [MultiAgentSystem (multi_agent.py)](#4-multiagentsystem-multi_agentpy)
5. [命令行聊天工具 (chat.py)](#5-命令行聊天工具-chatpy)

---

## 1. LLMClient (`llm_client.py`)

`LLMClient` 是一个通用的 LLM 客户端，支持 OpenAI 和 Google Gemini API。它对外提供统一的接口，屏蔽了不同供应商的底层差异。

### 类定义

```python
class LLMClient(
    api_key: str,
    base_url: str,
    model: str,
    provider: str = "openai",
    temperature: float = 0.7,
    max_tokens: int = 2000,
    **kwargs
)
```

### 初始化参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `api_key` | `str` | 是 | - | API 密钥。 |
| `base_url` | `str` | 是 | - | API 的基础 URL (例如 `https://api.openai.com/v1` 或自定义端点)。 |
| `model` | `str` | 是 | - | 使用的模型名称 (例如 `gpt-4`, `gemini-pro`)。 |
| `provider` | `str` | 否 | `"openai"` | 供应商类型。可选值: `"openai"`, `"google"`。 |
| `temperature` | `float` | 否 | `0.7` | 采样温度，控制输出的随机性。 |
| `max_tokens` | `int` | 否 | `2000` | 生成的最大 Token 数。 |
| `**kwargs` | `dict` | 否 | - | 其他传递给底层客户端的额外参数。 |

### 核心方法

#### 1. `chat(messages, **kwargs) -> str`

发送对话请求并获得完整响应。

*   **参数**:
    *   `messages` (`Iterable[ChatCompletionMessageParam]`): 消息列表。
    *   `**kwargs`: 覆盖初始化时的配置（如 `temperature`, `max_tokens`, `schema`）。
*   **返回**: `str` (模型生成的文本内容。如果使用了 `schema` 参数，返回符合 schema 的 JSON 字符串)。

#### 2. `stream_chat(messages, **kwargs) -> Iterator[str]`

流式发送对话请求，返回生成内容的迭代器。

*   **参数**: 同 `chat`。
*   **返回**: `Iterator[str]` (逐步产出的文本片段。如果使用了 `schema` 参数，完整响应为符合 schema 的 JSON 字符串)。

### 结构化输出 (Structured Output)

> **注意**: 此功能仅在 `provider="google"` 时可用。

结构化输出功能允许你指定模型输出的 JSON 格式，确保返回的数据符合预期的结构。这对于需要解析 LLM 输出的应用非常有用。

**无需导入 Google SDK！** 使用简单的字典格式定义 schema 即可。

#### 使用方法

```python
from llm_client import LLMClient
import json

# 1. 定义输出 Schema（使用字典格式）
person_schema = {
    "type": "OBJECT",
    "properties": {
        "name": {
            "type": "STRING",
            "description": "姓名"
        },
        "age": {
            "type": "INTEGER",
            "description": "年龄"
        },
        "hobbies": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "爱好列表"
        }
    },
    "required": ["name", "age"]
}

# 2. 创建 Google provider 客户端
client = LLMClient(
    api_key="your-google-api-key",
    base_url="https://generativelanguage.googleapis.com",
    model="gemini-2.0-flash",
    provider="google"
)

# 3. 调用 chat 并传入 schema
messages = [{"role": "user", "content": "介绍一下爱因斯坦"}]
response = client.chat(messages, schema=person_schema)

# 4. 解析 JSON 响应
data = json.loads(response)
print(data["name"])  # "Albert Einstein"
print(data["age"])   # 76
```

#### Schema 类型

支持以下数据类型（字符串格式）：

| 类型 | 说明 | 示例值 |
| :--- | :--- | :--- |
| `"STRING"` | 字符串 | `"hello"` |
| `"INTEGER"` | 整数 | `42` |
| `"NUMBER"` | 浮点数 | `3.14` |
| `"BOOLEAN"` | 布尔值 | `true` |
| `"ARRAY"` | 数组 | `["a", "b"]` |
| `"OBJECT"` | 对象 | `{"key": "value"}` |

#### Schema 属性说明

| 属性 | 说明 | 示例 |
| :--- | :--- | :--- |
| `type` | 数据类型 | `"STRING"`, `"OBJECT"`, `"ARRAY"` |
| `description` | 字段描述（可选） | `"用户的姓名"` |
| `properties` | OBJECT 类型的子字段定义 | `{"name": {"type": "STRING"}}` |
| `items` | ARRAY 类型的元素定义 | `{"type": "STRING"}` |
| `required` | 必填字段列表 | `["name", "age"]` |

#### 高级示例：嵌套结构

```python
# 定义嵌套的 Schema
company_schema = {
    "type": "OBJECT",
    "properties": {
        "company_name": {"type": "STRING"},
        "founded_year": {"type": "INTEGER"},
        "headquarters": {
            "type": "OBJECT",
            "properties": {
                "city": {"type": "STRING"},
                "country": {"type": "STRING"}
            },
            "required": ["city", "country"]
        },
        "products": {
            "type": "ARRAY",
            "items": {"type": "STRING"}
        }
    },
    "required": ["company_name", "founded_year"]
}
```

#### 从 YAML 文件加载 Schema

可以将 schema 定义在 YAML 文件中，方便管理和复用：

```yaml
# schema.yaml
schema:
  type: "OBJECT"
  properties:
    reasoning:
      type: "STRING"
      description: "推理过程说明"
    action:
      type: "ARRAY"
      description: "动作指令列表"
      items:
        type: "STRING"
        description: "单个动作描述"
  required:
    - "reasoning"
    - "action"
```

```python
import yaml
from llm_client import LLMClient

# 从 YAML 文件加载 schema
with open("schema.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    schema = config["schema"]

client = LLMClient(
    api_key="your-api-key",
    base_url="https://generativelanguage.googleapis.com",
    model="gemini-2.0-flash",
    provider="google"
)

response = client.chat(messages, schema=schema)
```

#### 流式输出

结构化输出同样支持流式响应：

```python
full_response = ""
for chunk in client.stream_chat(messages, schema=person_schema):
    print(chunk, end="", flush=True)
    full_response += chunk

# 流式完成后解析
data = json.loads(full_response)
```

#### 与 Conversation 类配合使用

```python
from llm_client import LLMClient
from conversation import Conversation

client = LLMClient(
    api_key="your-api-key",
    base_url="https://generativelanguage.googleapis.com",
    model="gemini-2.0-flash",
    provider="google"
)

conversation = Conversation(client, system_prompt="你是一个数据分析助手。")

sentiment_schema = {
    "type": "OBJECT",
    "properties": {
        "sentiment": {"type": "STRING", "description": "情感倾向"},
        "confidence": {"type": "NUMBER", "description": "置信度 0-1"}
    },
    "required": ["sentiment", "confidence"]
}

# schema 参数会通过 **kwargs 传递到底层 client
response = conversation.send("分析这段文本的情感", schema=sentiment_schema)
```

### 结构化输出 (Structured Output)

当使用 Google Gemini 作为供应商时 (`provider="google"`)，可以通过 `response_mime_type` 和 `response_schema` 参数来强制模型输出结构化的 JSON 数据。

#### 示例 1: 仅指定 JSON 格式

```python
response = client.chat(
    messages,
    response_mime_type="application/json"
)
```

#### 示例 2: 指定详细的 JSON Schema

你可以定义一个标准的 JSON Schema 对象来约束输出字段。

```python
# 定义 Schema
schema = {
    "type": "OBJECT",
    "properties": {
        "reasoning": {
            "type": "STRING",
            "description": "思考过程"
        },
        "action": {
            "type": "OBJECT",
            "properties": {
                "direction": {
                    "type": "STRING",
                    "enum": ["前", "后", "左", "右", "停"]
                },
                "speed": {
                    "type": "STRING",
                    "enum": ["low", "medium", "high"]
                }
            },
            "required": ["direction", "speed"]
        }
    },
    "required": ["reasoning", "action"]
}

# 调用
response = client.chat(
    messages,
    response_mime_type="application/json",
    response_schema=schema
)
```

**支持的 Schema 类型**:
*   `STRING`: 字符串
*   `NUMBER`: 数字 (float)
*   `INTEGER`: 整数
*   `BOOLEAN`: 布尔值
*   `ARRAY`: 数组 (需指定 `items`)
*   `OBJECT`: 对象 (需指定 `properties`)

#### 详细说明

1.  **MIME 类型 (`response_mime_type`)**:
    *   `"application/json"`: 最常用的格式，返回标准的 JSON 字符串。
    *   `"text/plain"`: 返回普通文本（默认）。
    *   `"text/x.enum"`: 返回枚举值（通常配合简单的 Schema 使用）。

2.  **Schema 格式规则 (`response_schema`)**:
    Schema 是一个标准的 Python 字典，遵循 OpenAPI 3.0 数据类型定义的子集。主要字段如下：

    *   `type` (必填): 数据类型，支持：
        *   `STRING`: 文本字符串。
        *   `NUMBER`: 浮点数。
        *   `INTEGER`: 整数。
        *   `BOOLEAN`: 布尔值 (`true`/`false`)。
        *   `ARRAY`: 列表/数组。
        *   `OBJECT`: 字典/对象。
    
    *   `description` (可选): 字段描述，用于指导模型理解字段含义（非常重要，相当于 Prompt 的一部分）。
    
    *   `nullable` (可选): 布尔值，是否允许为 `null`。
    
    *   `enum` (可选): 字符串列表，仅用于 `STRING` 类型，限制输出值为列表中的一项。
    
    *   `format` (可选): 字符串，指定格式（如 `"float"`, `"double"`, `"int32"`, `"int64"`）。

    *   `properties` (当 `type` 为 `OBJECT` 时必填): 字典，定义对象的属性（字段名 -> Schema）。

    *   `required` (当 `type` 为 `OBJECT` 时可选): 字符串列表，指定必须包含的属性名。

    *   `items` (当 `type` 为 `ARRAY` 时必填): 字典，定义数组元素的 Schema。

#### 复杂 Schema 示例

```python
complex_schema = {
    "type": "OBJECT",
    "properties": {
        "product_name": {"type": "STRING", "description": "产品名称"},
        "tags": {
            "type": "ARRAY",
            "description": "产品标签列表",
            "items": {"type": "STRING"}
        },
        "specs": {
            "type": "OBJECT",
            "description": "技术参数",
            "properties": {
                "weight": {"type": "NUMBER", "description": "重量(kg)"},
                "dimensions": {
                    "type": "OBJECT",
                    "properties": {
                        "width": {"type": "INTEGER"},
                        "height": {"type": "INTEGER"}
                    }
                }
            }
        }
    }
}
```

### 输入格式说明 (`messages`)

`messages` 是一个字典列表，每个字典包含 `role` 和 `content`：

```python
messages = [
    {"role": "system", "content": "你是一个有用的助手。"},
    {"role": "user", "content": "你好，请介绍一下自己。"}
]
```
*   **role**: 角色，通常为 `"system"`, `"user"`, `"assistant"`。
*   **content**: 消息内容字符串。

---

## 2. Conversation (`conversation.py`)

`Conversation` 类用于管理对话历史和上下文，它封装了 `LLMClient` 并自动维护消息列表。

### 类定义

```python
class Conversation(
    client: LLMClient,
    system_prompt: Optional[str] = None
)
```

### 初始化参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `client` | `LLMClient` | 是 | - | 已初始化的 LLM 客户端实例。 |
| `system_prompt` | `str` | 否 | `None` | 系统提示词，若提供则作为第一条消息。 |

### 核心方法

*   **`add_message(role: str, content: str)`**: 手动添加一条消息到历史记录。
*   **`send(user_message: str, **kwargs) -> str`**: 发送用户消息，自动保存用户提问和 AI 回复到历史，并返回 AI 回复。
*   **`stream_send(user_message: str, **kwargs) -> Iterator[str]`**: 同上，但是以流式返回。
*   **`clear()`**: 清空对话历史（如果初始化时有 `system_prompt`，会保留）。
*   **`get_history() -> List[Dict]`**: 获取当前的完整对话历史。
*   **`set_system_prompt(prompt: str)`**: 更新或设置系统提示词。

---

## 3. Agent (`agent.py`)

`Agent` 类用于构建具有特定角色、个性和背景的 AI 智能体。它在 `Conversation` 的基础上增加了角色扮演的系统提示词构建逻辑。

### 类定义

```python
class Agent(
    client: LLMClient,
    name: str = "Agent",
    role: str = "",
    personality: str = "",
    background: str = ""
)
```

### 初始化参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `client` | `LLMClient` | 是 | - | LLM 客户端实例。 |
| `name` | `str` | 否 | `"Agent"` | 智能体的名称。 |
| `role` | `str` | 否 | `""` | 角色设定 (例如 "一位经验丰富的医生")。 |
| `personality` | `str` | 否 | `""` | 性格特征 (例如 "温和、耐心、严谨")。 |
| `background` | `str` | 否 | `""` | 背景故事。 |

### 工作机制

初始化时，`Agent` 会自动调用 `_build_system_prompt()` 方法，将 `role`, `personality`, `background` 组合成一个强约束的 System Prompt，要求模型：
1.  **严禁破格**：不透露自己是 AI。
2.  **第一人称**：始终以设定角色的身份回答。

### 核心方法

*   **`respond(message: str, **kwargs) -> str`**: 代理调用 `conversation.send`。
*   **`stream_respond(message: str, **kwargs)`**: 代理调用 `conversation.stream_send`。
*   **`reset()`**: 重置记忆。
*   **`get_history()`**: 获取记忆。

---

## 4. MultiAgentSystem (`multi_agent.py`)

该模块包含 `MultiAgentSystem` 基类和 `WerewolfGame` 示例类，用于管理多个 Agent 之间的交互。

### `MultiAgentSystem` 类

用于管理一组 Agent，支持群发消息和轮询对话。

#### 核心方法

*   **`add_agent(agent: Agent)`**: 注册一个 Agent。
*   **`remove_agent(name: str)`**: 移除一个 Agent。
*   **`get_agent(name: str) -> Agent`**: 获取指定名称的 Agent。
*   **`broadcast(message: str, exclude: List[str] = None) -> Dict[str, str]`**:
    *   向所有 Agent (除了 `exclude` 列表中的) 发送同一条消息。
    *   返回字典 `{agent_name: response_text}`。
*   **`round_robin(initial_message: str, rounds: int = 1) -> List[Dict[str, str]]`**:
    *   让 Agent 依次发言，前一个 Agent 的输出会作为下一个 Agent 的输入的一部分。
    *   返回每一轮的对话记录列表。
*   **`reset_all()`**: 重置所有 Agent 的记忆。

### `WerewolfGame` 类

继承自 `MultiAgentSystem`，展示了如何扩展多智能体系统来维护游戏状态。

*   **额外属性**: `game_state` (包含 `phase`, `day`, `alive_players` 等)。
*   **特定方法**: `start_game`, `night_phase`, `day_phase`, `eliminate_player`。

---

## 5. 命令行聊天工具 (`chat.py`)

这是一个可直接运行的脚本，用于在终端与 LLM 进行交互。

### 使用方法

1.  确保项目根目录下存在 `config.yaml` 配置文件。
2.  配置文件格式示例：
    ```yaml
    api:
      api_key: "your-api-key"
      base_url: "https://api.openai.com/v1"
      model: "gpt-3.5-turbo"
      temperature: 0.7
      max_tokens: 2000
    ```
3.  运行命令：
    ```bash
    python chat.py
    ```

### 交互指令
*   输入文字进行对话。
*   输入 `clear` 清空当前对话历史。
*   输入 `quit` 退出程序。
