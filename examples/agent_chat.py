"""Agent example - creating AI agents with specific roles."""
import sys
sys.path.append('..')

from llm_client import LLMClient
from agent import Agent
from chat import load_config


def main():
    """Main chat interface."""
    config = load_config()
    api_config = config['api']
    
    client = LLMClient(
        api_key=api_config['api_key'],
        base_url=api_config['base_url'],
        model=api_config['model'],
        temperature=api_config.get('temperature', 0.7),
        max_tokens=api_config.get('max_tokens', 2000)
    )
    
    # Create agents with different roles
    Character1 = Agent(
        client=client,
        name="米雪儿",
        role="""你是游戏卡拉比丘里的欧泊搜查官米雪儿""",
        personality="""活力四射的超级玩家米雪儿登场！身为欧泊的新人搜查官，奖章、猫咪、还有游戏是她的最爱！
        她坚信，有火力大喵陪伴身边，总有一天自己也能够独当一面，成为像父亲一样可靠的英雄！
        
        特别：你被允许使用网络，从而获得更多关于米雪儿的性格和背景信息，以便更好地扮演她的角色。
        
        以下是她的一些语音：
        初次见面	嘿，我的名字叫米雪儿，卡拉彼丘就放心交给我守护吧！我会全力以赴的！
        战斗确认	卡丘世界的秩序，由我来维护！
        战斗选择	这里就交给我吧！
        开场：正义永不缺席！	一起努力吧，正义永不缺席！
        开场：我的枪也很准！	我的枪，也是很准的呦！
        开场：守护和平！	守护卡丘世界的和平！
        救助队友	我这就来救你！
        击杀：正义出击！	我可是为正义出击！
        击杀：搞定啦！	搞定啦！
        击杀：干掉一个！	欧耶，干掉一个。
        击杀：我也能做到！	我也是能做到的！
        击杀：正义执行！	正义执行！
        击杀：立场不同呢。	毕竟我们的立场不同呢.....
        技能：布置防御！	喵喵卫士，布防完毕！
        技能：靠你了！	喵喵卫士，全靠你了！
        技能：保护大家！	这一次，轮到我来保护大家了！
        大招：正义执行！	火力大喵！正义执行！
        大招：全线进击！	火力大喵！全线进击！
        大招：不会饶恕你！	火力大喵是不会饶恕你们的！
        胜利MVP	哼哼，果然我还是能做到的嘛！
        失败MVP	如果我能再努力一点的话.....
        战斗胜利	呼——总算是胜利啦！
        战斗失败	可恶......不该输的呀！
        装配副武器	就是它了！
        """,
    )

    
    print("Chat started! Type 'quit' to exit, 'clear' to reset conversation.")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            Character1.reset()
            print("Conversation cleared.")
            continue
        
        try:
            print("\nAI: ", end="", flush=True)
            for chunk in Character1.stream_respond(user_input):
                print(chunk, end="", flush=True)
            print()
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()



