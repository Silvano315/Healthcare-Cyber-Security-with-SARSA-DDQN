import gym_idsgame
import gymnasium as gym
import numpy as np
import psutil
import os
import time

from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
from gymnasium.spaces import Discrete, Box
from collections import defaultdict


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

@dataclass
class RewardAnalysis:
    """Analysis of reward mechanisms.
    
    Attributes:
        successful_attack_reward: Reward for successful attacks
        failed_attack_penalty: Penalty for failed attacks
        defense_reward: Reward for successful defense
        step_penalty: Penalty for each step taken
    """
    successful_attack_reward: float
    failed_attack_penalty: float
    defense_reward: float
    step_penalty: float

@dataclass
class GameDynamics:
    """Information about game dynamics and strategies.
    
    Attributes:
        win_conditions: Conditions that lead to victory
        loss_conditions: Conditions that lead to loss
        average_episode_length: Average number of steps per episode
        common_attack_patterns: Frequently observed attack sequences
    """
    win_conditions: Dict[str, Any]
    loss_conditions: Dict[str, Any]
    average_episode_length: float
    common_attack_patterns: Dict[str, int]

@dataclass
class PerformanceMetrics:
    """Performance-related measurements.
    
    Attributes:
        avg_step_time: Average time per step
        memory_usage: Memory usage statistics
        episode_statistics: Statistics about episode duration and outcomes
    """
    avg_step_time: float
    memory_usage: Dict[str, float]
    episode_statistics: Dict[str, float]


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
        self.initial_obs, _ = self.env.reset(seed=None)

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
        try:

            for step in range(num_steps):
                attack_action = self.env.action_space.sample()
                defense_action = self.env.action_space.sample()
                action = (attack_action, defense_action)

                next_obs, reward, terminated, truncated, info = self.env.step(action)

                transition_info = {
                    "step" : step + 1,
                    "action" : {
                        "attack" : attack_action,
                        "defense" : defense_action
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
                    current_obs, _ = self.env.reset(seed=None)
                else:
                    current_obs = next_obs
        finally:
            self.env.close()
        self.env.close()
        return {"transitions": transitions}
        
    def analyze_reward_mechanism(self, num_episodes: int = 100) -> RewardAnalysis:
        """Analyze the reward mechanism through multiple episodes.
        
        Args:
            num_episodes: Number of episodes to analyze
            
        Returns:
            RewardAnalysis: Statistical analysis of rewards
        """
        rewards = {
            'successful_attack': [],
            'failed_attack': [],
            'defense': [],
            'step': []
        }

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_rewards = []
            
            while not done:
                attack_action = self.env.action_space.sample()
                defense_action = self.env.action_space.sample()
                action = (attack_action, defense_action)
                
                _, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_rewards.append(reward)
        
                if isinstance(reward, tuple):
                    attack_reward, defense_reward = reward
                    if attack_reward > 0:
                        rewards['successful_attack'].append(attack_reward)
                    elif attack_reward < 0:
                        rewards['failed_attack'].append(attack_reward)
                    rewards['defense'].append(defense_reward)
                else:
                    rewards['step'].append(reward)
        
        return RewardAnalysis(
            successful_attack_reward=np.mean(rewards['successful_attack']) if rewards['successful_attack'] else 0,
            failed_attack_penalty=np.mean(rewards['failed_attack']) if rewards['failed_attack'] else 0,
            defense_reward=np.mean(rewards['defense']) if rewards['defense'] else 0,
            step_penalty=np.mean(rewards['step']) if rewards['step'] else 0
        )
    
    def analyze_game_dynamics(self, num_episodes: int = 100) -> GameDynamics:
        """Analyze game dynamics through multiple episodes.
        
        Args:
            num_episodes: Number of episodes to analyze
            
        Returns:
            GameDynamics: Analysis of game dynamics and patterns
        """
        episode_lengths = []
        attack_patterns = defaultdict(int)
        win_conditions = defaultdict(int)
        loss_conditions = defaultdict(int)

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            steps = 0
            current_attack_pattern = []
            
            while not done:
                attack_action = self.env.action_space.sample()
                defense_action = self.env.action_space.sample()
                action = (attack_action, defense_action)
                
                _, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                steps += 1
                
                current_attack_pattern.append(attack_action)
                
                if done:
                    if isinstance(reward, tuple):
                        attack_reward, _ = reward
                        condition = 'win' if attack_reward > 0 else 'loss'
                        if condition == 'win':
                            win_conditions['attack_successful'] += 1
                        else:
                            loss_conditions['attack_failed'] += 1
            
            episode_lengths.append(steps)
            if len(current_attack_pattern) > 2:
                pattern_key = tuple(current_attack_pattern[-3:])  
                attack_patterns[pattern_key] += 1
        
        return GameDynamics(
            win_conditions=dict(win_conditions),
            loss_conditions=dict(loss_conditions),
            average_episode_length=np.mean(episode_lengths),
            common_attack_patterns=dict(sorted(attack_patterns.items(), 
                                             key=lambda x: x[1], 
                                             reverse=True)[:5]) 
        )

    def analyze_performance(self, num_steps: int = 1000) -> PerformanceMetrics:
        """Analyze performance characteristics of the environment.
        
        Args:
            num_steps: Number of steps to analyze
            
        Returns:
            PerformanceMetrics: Performance measurements
        """
        step_times = []
        memory_usage = []
        process = psutil.Process(os.getpid())
        
        obs, _ = self.env.reset()
        
        for _ in range(num_steps):
            start_time = time.time()
            
            attack_action = self.env.action_space.sample()
            defense_action = self.env.action_space.sample()
            action = (attack_action, defense_action)
            
            _, _, terminated, truncated, _ = self.env.step(action)
            
            step_times.append(time.time() - start_time)
            memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
            
            if terminated or truncated:
                obs, _ = self.env.reset()
        
        return PerformanceMetrics(
            avg_step_time=np.mean(step_times),
            memory_usage={
                'mean': np.mean(memory_usage),
                'max': np.max(memory_usage),
                'min': np.min(memory_usage)
            },
            episode_statistics={
                'steps_per_second': 1.0 / np.mean(step_times),
                'memory_growth': memory_usage[-1] - memory_usage[0]
            }
        )

    def close(self):
        """Clean up environment resources."""
        self.env.close()


def main():
    """Main function to demonstrate the explorer's capabilities."""

    env_names = ["idsgame-random-attack-v0", "idsgame-maximal-attack-v0"]
    
    for env_name in env_names:
        print(f"\nTesting environment: {env_name}")
        try:
            explorer = IdsGameExplorer(env_name)
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
        except ValueError as e:
            print(f"Error with environment {env_name}: {e}")
        except Exception as e:
            print(f"Unexpected error with environment {env_name}: {e}")    

if __name__ == "__main__":
    main()