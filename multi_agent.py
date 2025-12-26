"""Multi-agent system for coordinating multiple AI agents."""
from typing import List, Dict, Callable, Optional
from agent import Agent


class MultiAgentSystem:
    """Manages multiple agents and their interactions."""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
    
    def add_agent(self, agent: Agent):
        """Add an agent to the system."""
        self.agents[agent.name] = agent
    
    def remove_agent(self, name: str):
        """Remove an agent from the system."""
        if name in self.agents:
            del self.agents[name]
    
    def get_agent(self, name: str) -> Optional[Agent]:
        """Get an agent by name."""
        return self.agents.get(name)
    
    def broadcast(self, message: str, exclude: List[str] = None) -> Dict[str, str]:
        """Send message to all agents and collect responses."""
        exclude = exclude or []
        responses = {}
        for name, agent in self.agents.items():
            if name not in exclude:
                responses[name] = agent.respond(message)
        return responses
    
    def round_robin(self, initial_message: str, rounds: int = 1) -> List[Dict[str, str]]:
        """Have agents respond in sequence for multiple rounds."""
        history = []
        current_message = initial_message
        
        for round_num in range(rounds):
            round_responses = {}
            for name, agent in self.agents.items():
                response = agent.respond(current_message)
                round_responses[name] = response
                current_message = f"{name}: {response}"
            history.append(round_responses)
        
        return history
    
    def reset_all(self):
        """Reset all agents' conversation histories."""
        for agent in self.agents.values():
            agent.reset()


class WerewolfGame(MultiAgentSystem):
    """Specialized multi-agent system for Werewolf game."""
    
    def __init__(self):
        super().__init__()
        self.game_state = {
            "phase": "night",
            "day": 0,
            "alive_players": [],
            "dead_players": []
        }
    
    def start_game(self):
        """Initialize game state."""
        self.game_state["day"] = 1
        self.game_state["phase"] = "night"
        self.game_state["alive_players"] = list(self.agents.keys())
        self.game_state["dead_players"] = []
    
    def night_phase(self) -> Dict[str, str]:
        """Execute night phase actions."""
        self.game_state["phase"] = "night"
        prompt = f"Night {self.game_state['day']}. What is your action?"
        return self.broadcast(prompt)
    
    def day_phase(self) -> Dict[str, str]:
        """Execute day phase discussions."""
        self.game_state["phase"] = "day"
        prompt = f"Day {self.game_state['day']}. Discuss and vote who you think is the werewolf."
        return self.broadcast(prompt)
    
    def eliminate_player(self, player_name: str):
        """Remove a player from the game."""
        if player_name in self.game_state["alive_players"]:
            self.game_state["alive_players"].remove(player_name)
            self.game_state["dead_players"].append(player_name)
    
    def next_day(self):
        """Advance to next day."""
        self.game_state["day"] += 1
