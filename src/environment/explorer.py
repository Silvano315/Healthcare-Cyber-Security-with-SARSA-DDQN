import gym_isdgame
import gymnasium as gym
import numpy as np

from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
from gymnasium.spaces import Discrete, Box

IDSGAME_ENV_MAPPING = {
    "idsgame-random-attack-v0": "gym_idsgame.envs.idsgame_env:IdsGameRandomAttackV0Env",
    "idsgame-maximal-attack-v0": "gym_idsgame.envs.idsgame_env:IdsGameMaximalAttackV0Env",
    "idsgame-minimal-defense-v0": "gym_idsgame.envs.idsgame_env:IdsGameMinimalDefenseV0Env",
    "idsgame-random-defense-v0": "gym_idsgame.envs.idsgame_env:IdsGameRandomDefenseV0Env",
    # Add other environment variants as needed
}

@dataclass
class EnvironmentMetrics:
    """Stores key metrics and properties of the IdsGame environment.
    
    Attributes:
        num_nodes (int): Number of nodes in the network
        num_attack_types (int): Number of different attack types available
        max_value (int): Maximum value possible in observation space
        defense_position (np.ndarray): Current defense agent position
        attack_position (np.ndarray): Current attack agent position
    """
    num_nodes: int
    num_attack_types: int
    max_value: int
    defense_position: np.ndarray
    attack_position: np.ndarray

class IdsGameExplorer:
    """A class to explore and analyze the IdsGame environment.
    
    This explorer helps understand the environment's structure, action space,
    observation space, and provides utilities for investigating state transitions.
    """
    
    def __init__(self, env_name: str = "idsgame-random-attack-v0"):
        """Initialize the explorer with a specific environment.
        
        Args:
            env_name (str): Name of the IdsGame environment variant to explore
            
        Raises:
            ValueError: If the environment name is not recognized
        """
        if env_name not in IDSGAME_ENV_MAPPING:
            raise ValueError(f"Unknown environment: {env_name}. Available environments: {list(IDSGAME_ENV_MAPPING.keys())}")
            
        if env_name not in gym.envs.registry:
            gym.register(
                id=env_name,
                entry_point=IDSGAME_ENV_MAPPING[env_name],
                kwargs={"idsgame_config": None, "save_dir": None, "initial_state_path": None}
            )

        self.env_name = env_name
        self.env = gym.make(env_name)
        self.initial_obs, _ = self.env.reset()

    def get_environment_metrics(self) -> EnvironmentMetrics:
        """Extract key metrics about the environment structure.
        
        Returns:
            EnvironmentMetrics: Dataclass containing environment properties
        """
        metrics = EnvironmentMetrics(
            num_nodes=self.env.idsgame_config.game_config.num_nodes,
            num_attack_types=self.env.idsgame_config.game_config.num_attack_types,
            max_value=self.env.idsgame_config.game_config.max_value,
            defense_position=self.initial_obs[0], 
            attack_position=self.initial_obs[1] 
        )
        return metrics
    
    def explore_action_space(self) -> Dict[str, Any]:
        """Analyze the structure and constraints of the action space.
        
        Returns:
            Dict[str, Any]: Dictionary containing action space properties
        """
        action_info = {
            "type" : type(self.env.action_space),
            "shape" : self.env.action_space.shape if hasattr(self.env.action_space, 'shape') else None,
            "sample" : self.env.action_space.sample(),
        }
        if isinstance(self.env.action_space, Discrete):
            action_info["n"] = self.env.action_space.n
        elif isinstance(self.env.action_space, Box):
            action_info.update({
                "low" : self.env.action_space.low,
                "high" : self.env.action_space.high,
            })
        return action_info
    
    def analyze_state_transition(self, 
                               num_steps: int = 1,
                               render: bool = True) -> Dict[str, Any]:
        """Analyze state transitions through RANDOM actions.
        
        Args:
            num_steps (int): Number of steps to analyze
            render (bool): Whether to render the environment
            
        Returns:
            Dict[str, Any]: Information about state transitions
        """
        transitions = []
        current_obs = self.initial_obs

        for step in range(num_steps):
            attack_action = self.env.action_space.sample()
            defence_action = self.env.action_space.sample()
            action = (attack_action, defence_action)

            next_obs, reward, terminated, truncated, info = self.env.step(action)

            transition_info = {
                "step" : step + 1,
                "action" : {
                    "attack" : attack_action,
                    "defence" : defence_action
                },
                "reward" : reward,
                "terminated" : terminated,
                "truncated" : truncated,
                "info" : info,
                "observation_change" : np.any(next_obs != current_obs)
            }
            transitions.append(transition_info)

            if render:
                try:
                    self.env.render()
                except Exception as e:
                    print(f"Rendering failed: {e}")

            if terminated or truncated:
                current_obs, _ = self.env.reset()
            else:
                current_obs = next_obs

        return {"transitions": transitions}

    def close(self):
        """Clean up environment resources."""
        self.env.close()

def main():
    """Main function to demonstrate the explorer's capabilities."""
    explorer = IdsGameExplorer

    try:
        metrics = explorer.get_environment_metrics()
        print("\n=== Environment Metrics ===")
        print(f"Number of nodes: {metrics.num_nodes}")
        print(f"Number of attack types: {metrics.num_attack_types}")
        print(f"Maximum value: {metrics.max_value}")
        print(f"Defense position shape: {metrics.defense_position.shape}")
        print(f"Attack position shape: {metrics.attack_position.shape}")

        action_info = explorer.explore_action_space()
        print("\n=== Action Space Analysis ===")
        for key, value in action_info.items():
            print(f"{key}: {value}")

        transition_info = explorer.analyze_state_transition(num_steps=3)
        print("\n=== State Transitions ===")
        for transition in transition_info["transitions"]:
            print(f"\nStep {transition['step']}:")
            print(f"Actions: {transition['action']}")
            print(f"Reward: {transition['reward']}")
            print(f"Terminated: {transition['terminated']}")
            print(f"State changed: {transition['observation_change']}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        explorer.close()

if __name__ == "__main__":
    main()