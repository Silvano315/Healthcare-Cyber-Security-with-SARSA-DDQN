import gymnasium as gym
import numpy as np
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

class IdsGameAnalyzer:
    """
    Analyzer for IdsGame environment focused on defense against random attacks
    """
    
    def __init__(self, env_name: str = "idsgame-random_attack-v9", entry_point: str = "IdsGameRandomAttackV9Env"):
        """
        Initialize analyzer with environment
        
        Args:
            env_name: Name of the IdsGame environment to analyze
            entry_point: Name of the entry point for the IdsGame environment
        """
        gym.register(
                id=env_name,
                entry_point='gym_idsgame.envs:' + entry_point,
                kwargs={"idsgame_config": None, "save_dir": None, "initial_state_path": None}
            )
        self.env = gym.make(env_name)
        self.env_name = env_name

    def inspect_env_source(self):
        """
        Print the available methods and attributes
        """
        print("Environment class:", self.env.unwrapped.__class__.__module__)
        print("\nAvailable methods and attributes:")
        for item in dir(self.env.unwrapped):
            if not item.startswith('_'):
                print(item)
        
        if hasattr(self.env.unwrapped, 'idsgame_config'):
            print("\nIDSGame Config:")
            print(self.env.unwrapped.idsgame_config)
    def analyze_observation_structure(self, num_steps: int = 5) -> Dict[str, Any]:
        """
        Analyze the structure of observations by collecting samples
        
        Args:
            num_steps: Number of steps to analyze
            
        Returns:
            Dictionary with observation analysis
        """
        obs, _ = self.env.reset()
        observations = []
        actions = []
        rewards = []
        
        print("\nInitial Observation Shape:", obs.shape)
        print("Initial Observation:\n", obs)
        
        for i in range(num_steps):
            defense_action = self.env.defender_action_space.sample()
            action = (-1, defense_action)
            next_obs, reward, done, _, info = self.env.step(action)
            
            print(f"\nStep {i+1}")
            print(f"Defense Action: {defense_action}")
            print(f"Reward: {reward}")
            print(f"Observation:\n", next_obs)
            print(f"Info: {info}")
            
            observations.append(next_obs)
            actions.append(action)
            rewards.append(reward)
            
            if done:
                break
                
        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards
        }
    
    def analyze_reward_structure(self, num_episodes: int = 5) -> Dict[str, List[float]]:
        """
        Analyze the reward structure by tracking different reward components
        
        Args:
            num_episodes: Number of episodes to analyze
            
        Returns:
            Dictionary with reward statistics
        """
        reward_stats = {
            "total_rewards": [],
            "hack_rewards": [],
            "detection_rewards": [],
            "blocked_rewards": []
        }
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            print(f"\nEpisode {episode+1}")
            step = 0
            
            while not done:
                defense_action = self.env.defender_action_space.sample()
                action = (-1, defense_action)
                next_obs, reward, done, _, info = self.env.step(action)
                episode_reward += reward
                
                # Track reward components
                hack_reward = self.env.get_hack_reward()
                detect_reward = self.env.get_detect_reward()
                blocked_reward = self.env.get_blocked_attack_reward()
                
                print(f"\nStep {step+1}")
                print(f"Total Reward: {reward}")
                print(f"Hack Reward: {hack_reward}")
                print(f"Detection Reward: {detect_reward}")
                print(f"Blocked Attack Reward: {blocked_reward}")
                
                step += 1
            
            reward_stats["total_rewards"].append(episode_reward)
            reward_stats["hack_rewards"].append(hack_reward)
            reward_stats["detection_rewards"].append(detect_reward)
            reward_stats["blocked_rewards"].append(blocked_reward)
            
        return reward_stats
    
    def visualize_state_transitions(self, num_steps: int = 5):
        """
        Visualize how states change with different defense actions
        """
        obs, _ = self.env.reset()
        print("\nAnalyzing State Transitions:")
        
        for i in range(num_steps):
            print(f"\nStep {i+1}")
            
            # Try each possible defense action
            for defense_action in range(self.env.num_defense_actions):
                test_obs, _ = self.env.reset()
                action = (-1, defense_action)
                next_obs, reward, done, _, info = self.env.step(action)
                
                print(f"\nDefense Action {defense_action}:")
                print(f"Reward: {reward}")
                print(f"Done: {done}")
                print(f"State Change: {np.sum(np.abs(next_obs - test_obs))}") 
                
                if done:
                    break
    
    def get_state_action_pairs(self, num_samples: int = 1000) -> List[Tuple[np.ndarray, int]]:
        """
        Generate sample state-action pairs for analysis
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            List of (state, action) tuples
        """
        samples = []
        obs, _ = self.env.reset()
        
        for _ in range(num_samples):
            defense_action = self.env.defender_action_space.sample()
            samples.append((obs.copy(), defense_action))
            
            action = (-1, defense_action)
            next_obs, reward, done, _, info = self.env.step(action)
            obs = next_obs
            
            if done:
                obs, _ = self.env.reset()
                
        return samples

    def analyze_spaces(self) -> Dict[str, Any]:
        """
        Analyze observation and action spaces

        Returns:
            Dictionary with observation and action insights
        """
        return {
            "env_name": self.env_name,
            "observation_space": {
                "shape": self.env.observation_space.shape,
                "type": str(self.env.observation_space),
            },
            "defender_action_space": {
                "n": self.env.num_defense_actions,
                "type": str(self.env.defender_action_space)
            },
            "attacker_action_space": {
                "n": self.env.num_attack_actions,
                "type": str(self.env.attacker_action_space)
            }
        }
    
    def analyze_game_parameters(self) -> Dict[str, Any]:
        """
        Analyze game-specific parameters and configuration

        Returns:
            Dictionary with features, actions and states information
        """
        return {
            "fully_observed": self.env.fully_observed,
            "local_view_features": self.env.local_view_features(),
            "reconnaissance_actions": self.env.is_reconnaissance(),
            "num_states": self.env.num_states,
            "num_states_full": self.env.num_states_full
        }
    
    def test_random_defense(self, num_episodes: int = 5, render: bool = False) -> Dict[str, List[float]]:
        """
        Test random defense actions against random attacks
        
        Args:
            num_episodes: Number of episodes to run
            render: Whether to render the environment
            
        Returns:
            Dictionary with episode statistics
        """
        stats = {
            "rewards": [],
            "hack_probability": [],
            "attack_detections": [],
            "steps": [],
            "failed_attacks": []
        }
        
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            while not done:
                if render:
                    self.env.render()
                
                # Random defense action
                defense_action = self.env.defender_action_space.sample()
                action = (-1, defense_action)  # -1 triggers random attack
                
                obs, reward, done, _, info = self.env.step(action)
                episode_reward += reward
                steps += 1
            
            stats["rewards"].append(episode_reward)
            stats["hack_probability"].append(self.env.hack_probability)
            stats["attack_detections"].append(self.env.attack_detections)
            stats["steps"].append(steps)
            stats["failed_attacks"].append(self.env.num_failed_attacks)
            
        return stats
    
    def analyze_defense_action(self, defense_action: int) -> Dict[str, Any]:
        """
        Analyze the effect of a specific defense action
        
        Args:
            defense_action: Defense action to analyze
            
        Returns:
            Dictionary with action analysis
        """
        obs, _ = self.env.reset()
        action = (-1, defense_action)
        
        # Check if defense is legal
        is_legal = self.env.is_defense_legal(defense_action)
        
        next_obs, reward, done, _, info = self.env.step(action)
        
        return {
            "is_legal": is_legal,
            "reward": reward,
            "done": done,
            "info": info,
            "detection_status": self.env.attack_detections,
            "hack_probability": self.env.hack_probability
        }
    
    def visualize_episode_stats(self, stats: Dict[str, List[float]]):
        """
        Visualize statistics from random episodes
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        ax1.plot(stats["rewards"], label="Episode Rewards")
        ax1.set_title("Episode Rewards")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        
        ax2.plot(stats["hack_probability"], label="Hack Probability")
        ax2.set_title("Hack Probability")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Probability")
        
        ax3.plot(stats["attack_detections"], label="Attack Detections")
        ax3.set_title("Attack Detections")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Count")
        
        ax4.plot(stats["failed_attacks"], label="Failed Attacks")
        ax4.set_title("Failed Attacks")
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Count")
        
        plt.tight_layout()
        plt.show()

    def run_full_analysis(self):
        """
        Run a comprehensive analysis of the environment
        """
        print("=== IdsGame Environment Analysis ===")
        
        print("\n1. Basic Information:")
        print(f"Environment Name: {self.env_name}")
        print(f"Observation Space Shape: {self.env.observation_space.shape}")
        print(f"Number of Defense Actions: {self.env.num_defense_actions}")
        print(f"Number of Attack Actions: {self.env.num_attack_actions}")
        
        print("\n2. Analyzing Observation Structure:")
        self.analyze_observation_structure(num_steps=3)
        
        print("\n3. Analyzing Reward Structure:")
        reward_stats = self.analyze_reward_structure(num_episodes=2)
        
        print("\n4. Testing State Transitions:")
        self.visualize_state_transitions(num_steps=2)
        
        return {
            "observation_samples": self.analyze_observation_structure(num_steps=5),
            "reward_stats": reward_stats,
            "state_action_samples": self.get_state_action_pairs(num_samples=100)
        }