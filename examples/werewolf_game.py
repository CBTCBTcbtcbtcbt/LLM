"""Werewolf game example with multiple AI agents."""
import sys
sys.path.append('..')

from llm_client import create_client
from agent import WerewolfPlayer
from multi_agent import WerewolfGame
from chat import load_config

# Load API config and create client
_config = load_config()
_api = _config.get('api', {})
client = create_client(
    provider=_api.get('provider', 'openai'),
    api_key=_api.get('api_key', ''),
    model=_api.get('model', 'gpt-3.5-turbo'),
    base_url=_api.get('base_url', 'https://api.openai.com/v1'),
    temperature=_api.get('temperature', 0.7),
    max_tokens=_api.get('max_tokens', 2000)
)

# Create game
game = WerewolfGame()

# Add players
players = [
    WerewolfPlayer("Alice", client, "villager"),
    WerewolfPlayer("Bob", client, "werewolf"),
    WerewolfPlayer("Charlie", client, "seer"),
    WerewolfPlayer("Diana", client, "villager"),
]

for player in players:
    game.add_agent(player)

# Start game
game.start_game()
print("Game started!")
print(f"Players: {', '.join(game.game_state['alive_players'])}")
print("="*50)

# Night phase
print("\n=== NIGHT PHASE ===")
night_actions = game.night_phase()
for player, action in night_actions.items():
    print(f"{player}: {action}")

# Day phase
print("\n=== DAY PHASE ===")
day_discussions = game.day_phase()
for player, discussion in day_discussions.items():
    print(f"{player}: {discussion}")
    print("-"*50)
