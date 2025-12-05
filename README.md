# LLM API 交互项目

一个模块化的Python项目，用于通过API与大语言模型（LLM）进行交互。采用面向对象编程（OOP）原则，结构清晰，易于复用和扩展。

## 项目结构

```
LLM/
├── config.yaml              # API配置文件
├── llm_client.py           # 核心LLM客户端模块
├── conversation.py         # 对话管理模块
├── agent.py                # AI智能体基类
├── multi_agent.py          # 多智能体系统
├── chat.py                 # 主聊天界面
├── requirements.txt        # 依赖包
├── examples/               # 示例代码
│   ├── simple_chat.py     # 简单对话示例
│   ├── agent_example.py   # 智能体示例
│   └── werewolf_game.py   # 狼人杀游戏示例
└── README.md              # 项目文档
```

## 功能特性

- **模块化设计**：所有功能都做了模块化，方便复用
- **多提供商支持**：支持OpenAI、Anthropic等多个LLM提供商
- **对话管理**：自动管理对话历史和上下文
- **智能体系统**：创建具有特定角色和个性的AI智能体
- **多智能体协作**：支持多个智能体之间的交互
- **流式响应**：支持流式输出，实时显示AI回复
- **配置文件**：API设置存储在配置文件中，易于管理

## 安装

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置API密钥：
编辑 `config.yaml` 文件，填入你的API密钥：
```yaml
api:
  provider: "openai"
  api_key: "your-api-key-here"
  model: "gpt-3.5-turbo"
  base_url: "https://api.openai.com/v1"
```

## 使用方法

### 1. 简单对话

运行主聊天程序：
```bash
python chat.py
```

或使用代码：
```python
from llm_client import create_client
from conversation import Conversation

client = create_client(
    provider="openai",
    api_key="your-api-key",
    model="gpt-3.5-turbo"
)

conv = Conversation(client)
response = conv.send("你好！")
print(response)
```

### 2. 创建AI智能体

```python
from llm_client import create_client
from agent import Agent

client = create_client(provider="openai", api_key="your-key", model="gpt-3.5-turbo")

# 创建具有特定角色的智能体
teacher = Agent(
    name="Teacher",
    client=client,
    role="你是一位耐心的老师",
    personality="友好且鼓励学生"
)

response = teacher.respond("什么是光合作用？")
print(response)
```

### 3. 多智能体系统

```python
from multi_agent import MultiAgentSystem

system = MultiAgentSystem()
system.add_agent(agent1)
system.add_agent(agent2)

# 广播消息给所有智能体
responses = system.broadcast("讨论一下气候变化")

# 轮流对话
history = system.round_robin("开始讨论", rounds=3)
```

### 4. 狼人杀游戏

```python
from agent import WerewolfPlayer
from multi_agent import WerewolfGame

game = WerewolfGame()

# 添加玩家
game.add_agent(WerewolfPlayer("Alice", client, "villager"))
game.add_agent(WerewolfPlayer("Bob", client, "werewolf"))

# 开始游戏
game.start_game()
night_actions = game.night_phase()
day_discussions = game.day_phase()
```

## 核心模块说明

### llm_client.py
- `LLMClient`: 抽象基类
- `OpenAIClient`: OpenAI API客户端
- `AnthropicClient`: Anthropic API客户端
- `create_client()`: 工厂函数，根据提供商创建客户端

### conversation.py
- `Conversation`: 管理对话历史和上下文
- 支持添加消息、发送消息、流式响应
- 可设置系统提示词

### agent.py
- `Agent`: AI智能体基类
- `WerewolfPlayer`: 狼人杀游戏专用智能体
- 支持自定义角色和个性

### multi_agent.py
- `MultiAgentSystem`: 多智能体管理系统
- `WerewolfGame`: 狼人杀游戏系统
- 支持广播、轮流对话等交互模式

## 扩展性

项目设计考虑了扩展性，你可以：

1. **添加新的LLM提供商**：继承 `LLMClient` 类
2. **创建自定义智能体**：继承 `Agent` 类
3. **设计新的多智能体系统**：继承 `MultiAgentSystem` 类
4. **添加新的游戏或应用场景**：使用现有模块组合

## 示例

查看 `examples/` 目录下的示例代码：
- `simple_chat.py`: 基础对话示例
- `agent_example.py`: 智能体使用示例
- `werewolf_game.py`: 狼人杀游戏示例

## 注意事项

- 使用前请确保已正确配置API密钥
- 注意API调用的费用
- 流式响应需要网络连接稳定

## 许可

本项目仅供学习和研究使用。
