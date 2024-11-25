from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.environment.idsgame_wrapper import IDSEnvironment

class IDSGameExplorer:
    """
    Explorer class for analyzing the IDS Game environment in detail
    """
    
    def __init__(self):
        """Initialize the explorer with the wrapped environment"""
        self.env = IDSEnvironment()
        
    def explore_state_transitions(self, num_steps: int = 10) -> Dict[str, List]:
        """
        Explore how states change with different defense actions
        """
        transitions = []
        obs, _ = self.env.reset()
        
        for _ in range(num_steps):
            # Try each possible defense action
            for defense_action in range(self.env.num_defense_actions):
                if self.env.is_defense_legal(defense_action):
                    action = (-1, defense_action)
                    next_obs, reward, done, _, info = self.env.step(action)
                    
                    transition = {
                        "defense_action": defense_action,
                        "reward": reward,
                        "state_change": np.sum(np.abs(next_obs - obs)),
                        "detection": len(self.env.attack_detections) > 0,
                        "attack_success": len(self.env.attacks) > 0
                    }
                    transitions.append(transition)
                    
                    if done:
                        obs, _ = self.env.reset()
                    else:
                        obs = next_obs
                        
        return transitions
    
    def analyze_reward_distribution(self, num_episodes: int = 100) -> Dict[str, np.ndarray]:
        """
        Analyze the distribution of rewards for different strategies
        """
        rewards = {
            "random": [],
            "minimal": []
        }
        
        # Random defense strategy
        for _ in range(num_episodes):
            episode_data = self.env.run_episode(random_defense=True)
            rewards["random"].extend(episode_data["rewards"])
            
        # Minimal defense strategy    
        for _ in range(num_episodes):
            episode_data = self.env.run_episode(random_defense=False)
            rewards["minimal"].extend(episode_data["rewards"])
            
        return rewards
    
    def analyze_defense_patterns(self, num_episodes: int = 10) -> Dict[str, List]:
        """
        Analyze patterns in defense effectiveness
        """
        patterns = {
            "detection_by_step": [],
            "defense_success_rate": [],
            "vulnerability_exploitation": []
        }
        
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_detections = []
            episode_defenses = []
            
            while not done:
                defense_action = self.env.defender_action_space.sample()
                action = (-1, defense_action)
                obs, reward, done, _, info = self.env.step(action)
                
                # Track patterns
                episode_detections.append(len(self.env.attack_detections))
                if len(self.env.defenses) > 0:
                    episode_defenses.append(self.env.defenses[-1])
                    
            patterns["detection_by_step"].append(episode_detections)
            patterns["defense_success_rate"].append(
                len(self.env.attack_detections) / max(1, len(self.env.attacks))
            )
            
        return patterns
    
    def visualize_attack_defense_patterns(self, num_episodes: int = 10):
        """
        Visualize patterns in attacks and defenses
        """
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        success_rates = []
        for _ in range(num_episodes):
            episode_data = self.env.run_episode()
            success_rates.append(np.mean(episode_data["attack_success"]))
        plt.plot(success_rates)
        plt.title("Attack Success Rate over Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Success Rate")
        
        plt.subplot(132)
        patterns = self.analyze_defense_patterns(num_episodes)
        plt.hist(patterns["defense_success_rate"], bins=20)
        plt.title("Defense Success Rate Distribution")
        plt.xlabel("Success Rate")
        plt.ylabel("Count")
        
        plt.subplot(133)
        rewards = self.analyze_reward_distribution(num_episodes=10)
        plt.boxplot([rewards["random"], rewards["minimal"]], labels=["Random", "Minimal"])
        plt.title("Reward Distribution by Strategy")
        plt.ylabel("Reward")
        
        plt.tight_layout()
        plt.show()
        
    def run_comprehensive_exploration(self):
        """
        Run a comprehensive exploration of the environment
        """
        print("=== Starting Comprehensive Environment Exploration ===\n")
        
        # Basic Environment Analysis
        self.env.comprehensive_analysis()
        
        # State Transition Analysis
        print("\n=== State Transition Analysis ===")
        transitions = self.explore_state_transitions(num_steps=5)
        print(f"\nAnalyzed {len(transitions)} state transitions")
        print(f"Average state change magnitude: {np.mean([t['state_change'] for t in transitions]):.2f}")
        print(f"Detection rate: {np.mean([t['detection'] for t in transitions]):.2%}")
        
        # Defense Pattern Analysis
        print("\n=== Defense Pattern Analysis ===")
        patterns = self.analyze_defense_patterns()
        print(f"Average detection rate: {np.mean(patterns['defense_success_rate']):.2%}")
        
        # Visualizations
        print("\n=== Generating Visualizations ===")
        self.visualize_attack_defense_patterns()
        
        return {
            "transitions": transitions,
            "patterns": patterns
        }