from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gym_idsgame.envs.dao.node_type import NodeType
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
        
        self.env.comprehensive_analysis()
        
        print("\n=== State Transition Analysis ===")
        transitions = self.explore_state_transitions(num_steps=5)
        print(f"\nAnalyzed {len(transitions)} state transitions")
        print(f"Average state change magnitude: {np.mean([t['state_change'] for t in transitions]):.2f}")
        print(f"Detection rate: {np.mean([t['detection'] for t in transitions]):.2%}")
        
        print("\n=== Defense Pattern Analysis ===")
        patterns = self.analyze_defense_patterns()
        print(f"Average detection rate: {np.mean(patterns['defense_success_rate']):.2%}")
        
        print("\n=== Generating Visualizations ===")
        self.visualize_attack_defense_patterns()
        
        return {
            "transitions": transitions,
            "patterns": patterns
        }
    
    def analyze_network_structure(self) -> Dict[str, Any]:
        """
        Analyze the network structure including nodes, connections, and topology
        
        Returns:
            Dict containing network analysis information
        """
        network_config = self.env.idsgame_config.game_config.network_config
        
        return {
            "num_layers": self.env.idsgame_config.game_config.num_layers,
            "nodes_per_layer": self.env.idsgame_config.game_config.num_servers_per_layer,
            "total_nodes": len(network_config.node_list),
            "adjacency_matrix": network_config.adjacency_matrix.copy(),
            "start_position": network_config.start_pos,
            "data_position": network_config.data_pos,
            "connected_layers": network_config.connected_layers,
            "fully_observed": network_config.fully_observed
        }
    
    def analyze_state_action_spaces(self) -> Dict[str, Any]:
        """
        Analyze the dimensions and properties of state and action spaces
        
        Returns:
            Dict containing state and action space analysis
        """
        return {
            "state_space": {
                "num_states": self.env.num_states,
                "num_states_full": self.env.num_states_full,
                "observation_shape": self.env.observation_space.shape,
                "defender_observation_shape": self.env.defender_action_space.shape
            },
            "action_space": {
                "num_attack_actions": self.env.num_attack_actions,
                "num_defense_actions": self.env.num_defense_actions,
                "attack_types": self.env.idsgame_config.game_config.num_attack_types,
                "max_value": self.env.idsgame_config.game_config.max_value
            }
        }
    
    def analyze_vulnerabilities(self) -> Dict[str, Any]:
        """
        Analyze the distribution and properties of vulnerabilities across nodes
        
        Returns:
            Dict containing vulnerability analysis
        """
        state = self.env.state
        network_config = self.env.idsgame_config.game_config.network_config
        
        node_vulnerabilities = []
        for node_id in range(len(network_config.node_list)):
            if network_config.node_list[node_id] != NodeType.EMPTY.value:
                node_vulns = {
                    "node_id": node_id,
                    "position": network_config.get_node_pos(node_id),
                    "defense_values": state.defense_values[node_id].copy(),
                    "attack_values": state.attack_values[node_id].copy(),
                    "detection": state.defense_det[node_id],
                    "min_defense": float(state.defense_values[node_id].min()),
                    "max_defense": float(state.defense_values[node_id].max())
                }
                node_vulnerabilities.append(node_vulns)
        
        return {
            "node_vulnerabilities": node_vulnerabilities,
            "vulnerabilities_per_node": self.env.idsgame_config.game_config.num_vulnerabilities_per_node,
            "vulnerabilities_per_layer": self.env.idsgame_config.game_config.num_vulnerabilities_per_layer,
            "total_vulnerabilities": sum(1 for node in node_vulnerabilities 
                                      if node["min_defense"] < self.env.idsgame_config.game_config.max_value)
        }
    
    def analyze_defense_effectiveness(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Analyze the effectiveness of defenses against different attack types
        
        Args:
            num_episodes: Number of episodes to analyze
            
        Returns:
            Dict containing defense effectiveness analysis
        """
        defense_stats = {
            "by_attack_type": {},
            "by_node": {},
            "detection_effectiveness": []
        }
        
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            
            while not done:
                defense_action = self.env.defender_action_space.sample()
                action = (-1, defense_action)
                obs, reward, done, _, info = self.env.step(action)
                
                for attack in self.env.attacks:
                    attack_type = attack[1]
                    if attack_type not in defense_stats["by_attack_type"]:
                        defense_stats["by_attack_type"][attack_type] = {
                            "total": 0,
                            "blocked": 0,
                            "detected": 0
                        }
                    
                    defense_stats["by_attack_type"][attack_type]["total"] += 1
                    if attack in self.env.failed_attacks:
                        defense_stats["by_attack_type"][attack_type]["blocked"] += 1
                    
                for defense in self.env.defenses:
                    node_id = defense[0]
                    if node_id not in defense_stats["by_node"]:
                        defense_stats["by_node"][node_id] = {
                            "total_defenses": 0,
                            "successful_defenses": 0
                        }
                    defense_stats["by_node"][node_id]["total_defenses"] += 1
                    if defense[2]:  
                        defense_stats["by_node"][node_id]["successful_defenses"] += 1
            
            if len(self.env.attacks) > 0:
                detection_rate = len(self.env.attack_detections) / len(self.env.attacks)
                defense_stats["detection_effectiveness"].append(detection_rate)
        
        return defense_stats
    
    def render_environment(self, mode: str = 'human') -> None:
        """
        Render the environment using the native renderer
        
        Args:
            mode: Rendering mode ('human' or 'rgb_array')
        """
        self.env.render(mode=mode)
        
    def visualize_network_structure(self) -> None:
        """
        Visualize the network structure using native rendering and print topology information
        """
        network_info = self.analyze_network_structure()
        
        print("\n=== Network Structure Analysis ===")
        print(f"\nTopology:")
        print(f"- Layers: {network_info['num_layers']}")
        print(f"- Nodes per layer: {network_info['nodes_per_layer']}")
        print(f"- Total nodes: {network_info['total_nodes']}")
        print(f"- Connected layers: {network_info['connected_layers']}")
        print(f"\nPositions:")
        print(f"- Start position: {network_info['start_position']}")
        print(f"- Data position: {network_info['data_position']}")
        
        print("\nRendering network structure...")
        self.render_environment()