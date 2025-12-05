# LLM API 交互项目 — 开发者技术文档

面向：希望在该项目基础上进行二次开发、扩展新能力（新增提供商、定制智能体、构建多智能体系统）的工程师。

目标：系统阐述架构设计、核心模块、数据流、配置管理、扩展指南、运行示例、测试与调试、性能与安全等，帮助开发者快速理解并高质量地构建功能。

---

## 1. 背景与设计目标

本项目是一个模块化的 Python 工程，用于通过 API 与大语言模型（LLM）进行交互。项目遵循面向对象设计原则（OOP），在保证清晰性的同时，强调：
- 低耦合：客户端、对话、智能体、编排彼此独立。
- 可扩展：采用抽象基类与工厂函数，便于新增 LLM 提供商与场景。
- 易用性：提供简单的 CLI 聊天入口与多种示例。
- 可维护：类型标注、清晰的模块职责、简单直观的数据流。

---

## 2. 术语约定
- Provider（提供商）：LLM 服务商，例如 OpenAI、Anthropic。
- Client（客户端）：具体 Provider 的 API 调用封装，例如 `OpenAIClient`。
- Conversation（对话）：维护消息历史与上下文的会话容器。
- Agent（智能体）：封装角色与个性的行为体，基于 `Conversation` 与 `LLMClient`。
- MultiAgent（多智能体）：协调多个 Agent 的交互编排组件。
- Streaming（流式）：服务端按增量片段返回内容的模式，适合实时输出。

---

## 3. 项目结构

```
LLM/
├── config.yaml              # API 配置文件
├── llm_client.py           # 核心 LLM 客户端模块（抽象/具体实现）
├── conversation.py         # 对话管理模块（历史、系统提示、流式发送）
├── agent.py                # 智能体基类 + WerewolfPlayer 示例
├── multi_agent.py          # 多智能体系统 + WerewolfGame 示例
├── chat.py                 # CLI 聊天入口（基于配置）
├── requirements.txt        # 依赖包
├── examples/               # 示例代码
│   ├── simple_chat.py     # 简单对话示例
│   ├── agent_example.py   # 智能体示例
│   └── werewolf_game.py   # 狼人杀游戏示例
└── README.md              # 项目简要说明
```

运行环境与依赖：
- Python 3.9+（建议）
- 依赖：`requests>=2.31.0`, `pyyaml>=6.0.1`

---

## 4. 架构总览

核心构件与关系：
- `LLMClient`（抽象）定义通用接口：`chat()` 与 `stream_chat()`。
- 具体客户端：`OpenAIClient`、`AnthropicClient` 实现 HTTP 调用与流式解析。
- `Conversation` 管理消息历史与系统提示，提供 `send()` 与 `stream_send()`。
- `Agent` 封装角色与个性，内部持有 `Conversation`，提供 `respond()` / `stream_respond()`。
- `MultiAgentSystem` 协调多个 `Agent` 的交互（广播、轮询），`WerewolfGame` 为特化示例。
- `chat.py` 基于 `config.yaml` 组装客户端与会话，实现 CLI 交互与流式输出。

数据流（简化）：
1. 加载配置 → `create_client()` 创建指定 Provider 的客户端。
2. 创建 `Conversation` → 收发用户消息。
3. `Conversation.send()` 或 `stream_send()` 调用 `client.chat()` / `client.stream_chat()`。
4. 响应被追加到会话历史并返回/实时输出。

---

## 5. 模块详解与接口参考

### 5.1 llm_client.py

核心类与函数：

```python
class LLMClient(ABC):
    def __init__(self, api_key: str, model: str, **kwargs): ...
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str: ...
    @abstractmethod
    def stream_chat(self, messages: List[Dict[str, str]], **kwargs): ...

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo",
                 base_url: str = "https://api.openai.com/v1", **kwargs): ...
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str: ...
    def stream_chat(self, messages: List[Dict[str, str]], **kwargs): ...

class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229",
                 base_url: str = "https://api.anthropic.com", **kwargs): ...
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str: ...
    def stream_chat(self, messages: List[Dict[str, str]], **kwargs): ...

def create_client(provider: str, **config) -> LLMClient: ...
```

- 公共配置：
  - `api_key`: 服务商密钥
  - `model`: 模型名称
  - `temperature`: 默认 0.7（可在 `kwargs` 传入覆盖）
  - `max_tokens`: 默认 2000（可在 `kwargs` 传入覆盖）
  - `base_url`: 服务端基础 URL（对 OpenAI/Anthropic 都支持传入）

- OpenAI 请求：
  - URL: `POST {base_url}/chat/completions`
  - Headers: `Authorization: Bearer <api_key>`, `Content-Type: application/json`
  - Body: `{ model, messages, temperature, max_tokens }`
  - 响应解析：`response.json()['choices'][0]['message']['content']`

- OpenAI 流式：
  - 在请求体加 `stream: True`
  - 通过 `response.iter_lines()` 读取 SSE 风格的 `data: ...` 行，遇到 `data: [DONE]` 结束。
  - 增量片段在 `choices[0].delta.content`。

- Anthropic 请求：
  - URL: `POST {base_url}/v1/messages`
  - Headers: `x-api-key`, `anthropic-version: 2023-06-01`, `Content-Type: application/json`
  - Body: `{ model, messages, temperature, max_tokens }`
  - 响应解析：`response.json()['content'][0]['text']`

- Anthropic 流式：
  - 在请求体加 `stream: True`
  - 通过 SSE `data: ...` 行解析，`type == 'content_block_delta'` 时增量文本位于 `delta.text`。

- 错误处理：
  - 均调用 `response.raise_for_status()`，网络或服务端错误将抛出 `requests.HTTPError`。
  - 建议扩展：设置 `timeout`、重试、退避、错误分类与日志。

- 扩展新的 Provider：
  1. 继承 `LLMClient`，实现 `chat()` 与 `stream_chat()`。
  2. 在 `create_client()` 的 `clients` 映射中注册新类。
  3. 实现 Provider 特有的认证、URL、请求与流式解析。

消息格式（与 OpenAI 对齐）：
```json
{
  "role": "system|user|assistant",
  "content": "文本内容"
}
```

### 5.2 conversation.py

职责：维护会话历史与系统提示，提供阻塞与流式发送。

关键接口：
```python
class Conversation:
    def __init__(self, client: LLMClient, system_prompt: Optional[str] = None): ...
    def add_message(self, role: str, content: str): ...
    def send(self, user_message: str, **kwargs) -> str: ...
    def stream_send(self, user_message: str, **kwargs): ...  # 生成器，yield 增量
    def clear(self): ...  # 保留系统提示（如存在），清空其它消息
    def get_history(self) -> List[Dict[str, str]]: ...
    def set_system_prompt(self, prompt: str): ...
```

流式逻辑：
- 先追加用户消息 → 调用 `client.stream_chat(messages, **kwargs)` → 持续累计 `full_response` 并 `yield` chunk → 结束后将完整回复追加到历史。

### 5.3 agent.py

职责：基于角色与个性构建 `system_prompt`，对外暴露面向场景的响应接口。

关键接口：
```python
class Agent:
    def __init__(self, name: str, client: LLMClient, role: str = "", personality: str = ""): ...
    def respond(self, message: str, **kwargs) -> str: ...
    def stream_respond(self, message: str, **kwargs): ...
    def reset(self): ...
    def get_history(self): ...
```

系统提示构建：
- 由 `role` 与 `personality` 拼装，示例格式：
```
Role: 你是一位耐心的老师
Personality: 友好且鼓励学生
```

特化示例：
- `WerewolfPlayer(Agent)`：根据角色类型（werewolf、villager、seer、doctor、hunter）设定不同个性与提示。

### 5.4 multi_agent.py

职责：对多个 `Agent` 进行编排，支持广播与轮询对话。

关键接口：
```python
class MultiAgentSystem:
    def add_agent(self, agent: Agent): ...
    def remove_agent(self, name: str): ...
    def get_agent(self, name: str) -> Optional[Agent]: ...
    def broadcast(self, message: str, exclude: List[str] = None) -> Dict[str, str]: ...
    def round_robin(self, initial_message: str, rounds: int = 1) -> List[Dict[str, str]]: ...
    def reset_all(self): ...

class WerewolfGame(MultiAgentSystem):
    def start_game(self): ...
    def night_phase(self) -> Dict[str, str]: ...
    def day_phase(self) -> Dict[str, str]: ...
    def eliminate_player(self, player_name: str): ...
    def next_day(self): ...
```

说明：
- `round_robin` 会将当前轮次最后一个回复拼接为下一轮的 `current_message`，形成串联对话。
- `WerewolfGame` 维护 `game_state`（phase/day/alive/dead），以简单规则驱动阶段提示。

### 5.5 chat.py（CLI 入口）

职责：读取配置，创建客户端与会话，提供交互式命令行聊天。

关键逻辑：
```python
def load_config(config_path: str = "config.yaml") -> dict: ...

def main():
    config = load_config()
    api_config = config['api']
    client = create_client(...)
    conversation = Conversation(client)
    while True:
        user_input = input("You: ").strip()
        if user_input == 'quit': break
        if user_input == 'clear': conversation.clear(); continue
        for chunk in conversation.stream_send(user_input):
            print(chunk, end="", flush=True)
```

交互指令：
- `quit`: 退出
- `clear`: 清空对话（保留系统提示）

---

## 6. 配置管理

示例（见 README.md）：
```yaml
api:
  provider: "openai"
  api_key: "your-api-key-here"
  model: "gpt-3.5-turbo"
  base_url: "https://api.openai.com/v1"
  temperature: 0.7
  max_tokens: 2000
```

说明：
- `provider`: 必填，当前支持 `openai`、`anthropic`。
- `api_key`: 必填，建议通过环境变量或密钥管理工具注入。
- `model`: 必填，参照各 Provider 的模型列表。
- `base_url`: 选填，默认指向官方地址，企业/代理场景可自定义。
- `temperature`、`max_tokens`: 可在配置或调用时覆盖。

安全建议：
- 不要将明文 `api_key` 提交到版本库。
- 针对多环境，将 `config.yaml` 抽象为模板，并使用外部注入敏感变量。

---

## 7. 扩展指南

### 7.1 新增 LLM 提供商

步骤：
1. 新建类 `MyProviderClient(LLMClient)`，实现 `chat()` 与 `stream_chat()`。
2. 处理认证（Headers）、URL、请求体与响应解析（含流式）。
3. 在 `create_client()` 的 `clients` 中注册：`'myprovider': MyProviderClient`。

注意点：
- 流式返回通常是 SSE（Server-Sent Events）或自定义分段协议，需正确解析与错误处理。
- 统一消息格式（role/content）以复用 `Conversation` 与 `Agent`。

### 7.2 自定义 Agent 行为

思路：
- 继承 `Agent` 并重写 `__init__()` 或新增方法，构建更复杂的系统提示（如工具使用、风格约束）。
- 复用 `Conversation`：可在 `respond()` 前注入 `system` 或 `assistant` 提示，引导模型输出。

示例：
```python
class CodeReviewer(Agent):
    def __init__(self, name, client):
        super().__init__(name, client,
            role="你是一名严格的代码评审专家",
            personality="简洁、可执行建议优先")
```

### 7.3 扩展多智能体编排

- 替换 `round_robin` 为更复杂的调度（如基于话题路由、上下文聚合、裁判评分）。
- 在 `broadcast` 前后插入中间件（如内容过滤、记忆写入、工具调用）。
- 引入共享工作记忆（vector store/数据库）以实现更强协作。

---

## 8. 错误处理与健壮性建议

当前实现：
- HTTP 层使用 `response.raise_for_status()` 直接抛错。
- 流式解析遇到 `json.JSONDecodeError` 时跳过该行继续，对不规范服务返回具有容错性。

建议补强：
- 设置请求 `timeout`，并引入重试（指数退避）。
- 捕获并分类错误：网络错误、认证错误、参数错误、配额限制（429）等。
- 结构化日志（请求耗时、模型、token、错误码），便于运维与成本控制。
- 输入校验（messages 结构、role 合法性、max_tokens 范围）。

---

## 9. 性能、成本与并发

- 流式优点：降低首字延迟，提升交互体验；建议在 CLI 与 UI 中默认启用。
- `max_tokens` 与 `temperature` 对性能与成本影响明显，应按场景调优。
- 并发：当前未内置并发调度；服务端限流时需在上层加入队列与重试。
- 缓存：可在上层对常见系统提示与少量问答做缓存，降低成本。

---

## 10. 安全与合规

- API Key 管理：使用环境变量/密钥管理服务；避免日志泄露。
- 数据敏感性：避免将敏感输入/输出写入明文日志或持久化储存。
- 访问控制：若部署在服务器侧，确保网络与凭证权限边界清晰。
- 成本控制：限制最大轮次与 token；对异常循环输出做截断保护。

---

## 11. 测试与调试建议

- 单元测试：
  - 使用 `requests` 的 mocking/fake server 验证 `chat()` 与 `stream_chat()` 行为。
  - 对 `Conversation` 的历史操作（`add_message`、`clear`、`set_system_prompt`）做断言。
  - 对 `MultiAgentSystem.broadcast/round_robin` 的交互结果进行一致性测试。
- 集成测试：
  - 在沙箱/API 速率限制宽松环境下运行 `examples/`，确保端到端串通。
- 调试：
  - 增加可选的 `debug` 标志输出请求体与响应段（注意隐私）。

---

## 12. 示例讲解

### 12.1 `examples/simple_chat.py`
- 创建 `client` → 构造 `Conversation(system_prompt=...)` → 连续 `send()` 两次输出结果。
- 用法简洁，适合作为最小可用 Demo。

### 12.2 `examples/agent_example.py`
- 创建两个角色不同的 `Agent`（Teacher/Scientist），对同一问题生成不同风格的回答。
- 演示了角色与个性对系统提示的影响。

### 12.3 `examples/werewolf_game.py`
- 构造 `WerewolfGame`，加入多名玩家（不同角色），启动游戏并分别在夜间与白天阶段进行广播式交互。
- 展示了多智能体编排与特化场景设计。

---

## 13. 常见问题（FAQ）与坑位

- 为什么响应为空或报错？
  - 检查 `api_key` 是否有效、`model` 与 `base_url` 是否正确、网络是否可达。
- 流式输出乱码或停滞？
  - 核查服务端的 SSE 格式与解析逻辑，确认 `data: [DONE]` 终止条件。
- 会话越聊越长导致成本上升？
  - 定期 `clear()`，或实现截断策略（仅保留最近 N 轮）。
- 不同 Provider 的消息格式不完全一致？
  - 项目统一使用 `role/content` 结构，必要时在客户端中做转换适配。

---

## 14. 未来改进方向

- 引入统一的错误模型与重试策略（含退避与幂等设计）。
- 增加日志与可观测性（请求耗时、token 计量、成本报表）。
- 支持函数调用（Tool-Use）与外部工具集成（检索/执行器）。
- 多智能体的更高级编排（话题路由、记忆共享、角色分工与评审）。
- 提供 Web UI 与更丰富的示例（插件化场景）。

---

## 15. 版本与兼容性

- 依赖：`requests>=2.31.0`, `pyyaml>=6.0.1`
- Python：建议 3.9+；注意 `typing` 兼容性（`List/Dict/Optional`）。
- Provider：当前支持 OpenAI 与 Anthropic，其他需自行扩展并适配。

---

## 16. 附录：接口速查与代码片段

- 创建客户端（工厂）：
```python
client = create_client(
    provider="openai",
    api_key="<KEY>",
    model="gpt-3.5-turbo",
    base_url="https://api.openai.com/v1",
    temperature=0.7,
    max_tokens=2000,
)
```

- 会话阻塞式：
```python
conv = Conversation(client, system_prompt="You are a helpful assistant.")
print(conv.send("Hello!"))
```

- 会话流式：
```python
for chunk in conv.stream_send("Tell me a joke"):
    print(chunk, end="", flush=True)
```

- Agent：
```python
teacher = Agent(
    name="Teacher",
    client=client,
    role="你是一位耐心的老师",
    personality="友好且鼓励学生",
)
print(teacher.respond("什么是光合作用？"))
```

- 多智能体：
```python
system = MultiAgentSystem()
system.add_agent(teacher)
system.add_agent(Agent("Scientist", client, role="科学家", personality="严谨"))
print(system.broadcast("讨论一下气候变化"))
print(system.round_robin("开始讨论", rounds=2))
```

---

## 17. 运行与验证

- 安装依赖：`pip install -r requirements.txt`
- 配置密钥：编辑 `config.yaml`
- 启动 CLI：`python chat.py`
- 运行示例：`python examples/simple_chat.py` 等

---

## 18. 许可与声明

- 本项目仅供学习与研究使用。
- 使用第三方 LLM 时需遵循对应服务条款与合规要求。
