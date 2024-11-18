import numpy as np
from typing import Dict, Any, Tuple
import os
import json
from src.agents.base_agent import BaseAgent

class SarsaAgent(BaseAgent):
    """
    SARSA Agent implementation focused on network defense.
    Uses SARSA (State-Action-Reward-State-Action) algorithm to learn optimal defense policies.
    """
    
    def __init__(self, env_wrapper: Any, config: Dict[str, Any]):
        """
        Initialize the SARSA agent.
        
        Args:
            env_wrapper: Wrapped IdsGame environment
            config: Configuration dictionary from config.yaml
        """
        super().__init__(env_wrapper, config)
        
        sarsa_config = config['sarsa']
        self.learning_rate = sarsa_config['learning_rate']
        self.gamma = sarsa_config['gamma']
        self.epsilon = sarsa_config['epsilon']['start']
        self.epsilon_end = sarsa_config['epsilon']['end']
        self.epsilon_decay = sarsa_config['epsilon']['decay']
        
        self.initialize_q_table()
        
        self.defense_success_count = 0
        self.total_defense_attempts = 0
        
    def initialize_q_table(self):
        """Initialize Q-table based on state and action space."""
        state_shape = self.env.get_state_size()
        if isinstance(state_shape, tuple):
            state_size = state_shape[0]
        else:
            state_size = state_shape
            
        action_size = self.env.get_action_size()
        
        self.q_table = {}
        
    def state_to_key(self, state: np.ndarray) -> str:
        """
        Convert state array to hashable key for Q-table.
        
        Args:
            state: Current state observation
            
        Returns:
            String representation of state
        """
        if isinstance(state, tuple):
            defense_state = state[0]
            return str(defense_state.tobytes())
        return str(state.tobytes())
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select defense action using ε-greedy policy.
        
        Args:
            state: Current state observation
            training: Whether the agent is training or evaluating
            
        Returns:
            Selected defense action
        """
        if training and np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        state_key = self.state_to_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.env.get_action_size())
            
        return np.argmax(self.q_table[state_key])
        
    def train_step(self, 
                  state: np.ndarray,
                  action: int,
                  reward: float,
                  next_state: np.ndarray,
                  done: bool) -> Dict[str, float]:
        """
        Perform a single SARSA training step to improve defense policy.
        
        Args:
            state: Current state
            action: Defense action taken
            reward: Defense reward received
            next_state: Next state
            done: Whether episode is done
            
        Returns:
            Dictionary of training metrics
        """
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.env.get_action_size())
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.env.get_action_size())
        
        next_action = self.select_action(next_state, training=True)
        
        current_q = self.q_table[state_key][action]
        next_q = 0 if done else self.q_table[next_state_key][next_action]
        
        self.q_table[state_key][action] = current_q + \
            self.learning_rate * (reward + self.gamma * next_q - current_q)
        
        self.total_defense_attempts += 1
        if reward > 0:  
            self.defense_success_count += 1
            
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            
        metrics = {
            'epsilon': self.epsilon,
            'defense_success_rate': self.defense_success_count / self.total_defense_attempts 
                                  if self.total_defense_attempts > 0 else 0.0,
            'q_value': current_q
        }
        
        return metrics
        
    def save(self, path: str):
        """
        Save agent's Q-table and training state.
        
        Args:
            path: Path to save location
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_dict = {
            'q_table': {str(k): v.tolist() for k, v in self.q_table.items()},
            'epsilon': self.epsilon,
            'defense_success_count': self.defense_success_count,
            'total_defense_attempts': self.total_defense_attempts
        }
        
        with open(path, 'w') as f:
            json.dump(save_dict, f)
            
    def load(self, path: str):
        """
        Load agent's Q-table and training state.
        
        Args:
            path: Path to load from
        """
        with open(path, 'r') as f:
            load_dict = json.load(f)
            
        self.q_table = {str(k): np.array(v) for k, v in load_dict['q_table'].items()}
        self.epsilon = load_dict['epsilon']
        self.defense_success_count = load_dict['defense_success_count']
        self.total_defense_attempts = load_dict['total_defense_attempts']