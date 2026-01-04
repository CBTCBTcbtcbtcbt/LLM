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
    *   `**kwargs`: 覆盖初始化时的配置（如 `temperature`, `max_tokens`）。
    *   **(Google Gemini 专用)** `response_mime_type` (`str`): 指定响应的 MIME 类型，例如 `"application/json"`。
    *   **(Google Gemini 专用)** `response_schema` (`dict`): 定义响应的 JSON Schema，强制模型输出符合特定结构的 JSON 数据。
*   **返回**: `str` (模型生成的文本内容)。

#### 2. `stream_chat(messages, **kwargs) -> Iterator[str]`

流式发送对话请求，返回生成内容的迭代器。

*   **参数**: 同 `chat`。
*   **返回**: `Iterator[str]` (逐步产出的文本片段)。

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
