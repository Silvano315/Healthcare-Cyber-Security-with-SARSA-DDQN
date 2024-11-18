from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np

class BaseAgent(ABC):
    """
    Abstract base class for all defensive agents (SARSA and DDQN).
    Provides common interface and functionality for network defense agents.
    """
    
    def __init__(self, env_wrapper: Any, config: Dict[str, Any]):
        """
        Initialize the base defensive agent.
        
        Args:
            env_wrapper: Wrapped environment instance
            config: Configuration dictionary from config.yaml
        """
        self.env = env_wrapper
        self.config = config
        self.state_size = env_wrapper.get_state_size()
        self.action_size = env_wrapper.get_action_size()
        
        self.episode_rewards = []        
        self.episode_lengths = []         
        self.defense_successes = 0       
        self.defense_failures = 0         
        self.training_steps = 0         
    
    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select a defensive action given the current state.
        
        Args:
            state: Current state observation
            training: Whether the agent is training or evaluating
            
        Returns:
            Selected defensive action
        """
        pass
    
    @abstractmethod
    def train_step(self, 
                  state: np.ndarray,
                  action: int,
                  reward: float,
                  next_state: np.ndarray,
                  done: bool) -> Dict[str, float]:
        """
        Perform a single training step to improve defense policy.
        
        Args:
            state: Current state
            action: Defense action taken
            reward: Defense reward received (positive for successful defense)
            next_state: Next state
            done: Whether episode is done
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """
        Save the agent's defense policy and training state.
        
        Args:
            path: Path to save location
        """
        pass
    
    @abstractmethod
    def load(self, path: str):
        """
        Load the agent's defense policy and training state.
        
        Args:
            path: Path to load from
        """
        pass
    
    def update_metrics(self, reward: float, done: bool):
        """
        Update defense performance metrics.
        
        Args:
            reward: Defense reward received
            done: Whether episode is done
        """
        if reward > 0:
            self.defense_successes += 1
        else:
            self.defense_failures += 1
        
        if done:
            self.episode_lengths.append(self.training_steps)
            
    def get_defense_rate(self) -> float:
        """
        Calculate the current defense success rate.
        
        Returns:
            Float between 0 and 1 representing defense success rate
        """
        total_attempts = self.defense_successes + self.defense_failures
        if total_attempts == 0:
            return 0.0
        return self.defense_successes / total_attempts
    
    def log_metrics(self, episode: int, metrics: Dict[str, float]):
        """
        Log training metrics with focus on defense performance.
        
        Args:
            episode: Current episode number
            metrics: Dictionary of metrics to log
        """
        if episode % self.config['sarsa']['logging']['frequency'] == 0:
            metrics['defense_rate'] = self.get_defense_rate()
            
            metrics_str = ' '.join([f"{k}: {v:.3f}" for k, v in metrics.items()])
            print(f"Episode {episode}: {metrics_str}")
    
    def reset_metrics(self):
        """Reset all defense-related metrics."""
        self.episode_rewards = []
        self.episode_lengths = []
        self.defense_successes = 0
        self.defense_failures = 0
        self.training_steps = 0