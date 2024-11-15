import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional
from gymnasium import spaces

class IdsGameWrapper:
    """
    A wrapper for the IdsGame environment that standardizes the interface
    and handles preprocessing of states and rewards.
    """
    
    def __init__(self, env_name: str, env_point: str, config: Dict[str, Any]):
        """
        Initialize the wrapper with specific environment configuration.
        
        Args:
            env_name: Name of the IdsGame environment
            env_point: Name of the entry point of the IdsGame environment
            config: Configuration dictionary from config.yaml
        """
        gym.register(
                id=env_name,
                entry_point='gym_idsgame.envs:' + env_point,
                kwargs={"idsgame_config": None, "save_dir": None, "initial_state_path": None}
            )
        
        self.env = gym.make(env_name)
        self.config = config

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.env_parameters = {
            'num_nodes' : self.env.idsgame_config.game_config.num_nodes,
            'num_attack_types': self.env.idsgame_config.game_config.num_attack_types,
            'max_value': self.env.idsgame_config.game_config.max_value
        }

        self.episode_rewards = []
        self.current_episode_reward = 0.0

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment and return processed initial state.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Processed initial state and info dict
        """
        observation, info, = self.env.reset(seed = seed)
        return self.preprocess_state(observation), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment with preprocessing of state and reward.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)
        processed_state = self.preprocess_state(next_state)
        processed_reward = self.process_reward(reward)

        self.current_episode_reward += processed_reward
        if terminated or truncated:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0

        return processed_state, processed_reward, terminated, truncated, info
    
    def preprocess_state(self, state: np.ndarray) -> np.ndarray:
        """
        Preprocess the state observation.
        
        Args:
            state: Raw state from environment
            
        Returns:
            Processed state
        """
        if isinstance(state, tuple):
            defense_state, attack_state = state
            # normalization with max_value
            processed_defense = defense_state / self.env_parameters['max_value']
            processed_attack = attack_state / self.env_parameters['max_value']
            return (processed_defense, processed_attack)
        else:
            return state / self.env_parameters['max_value']
    
    def process_reward(self, reward: float) -> float:
        """
        Process the reward signal.
        
        Args:
            reward: Raw reward from environment
            
        Returns:
            Processed reward
        """
        if isinstance(reward, tuple):
            attack_reward, defense_reward = reward
            # Focus on attack reward for training
            return float(attack_reward)
        return float(reward)