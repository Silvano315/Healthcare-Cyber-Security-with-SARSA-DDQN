from typing import Dict, Any, List, Tuple
import numpy as np
from gym_idsgame.envs.idsgame_env import IdsGameRandomAttackV21Env
from gym_idsgame.envs.dao.game_config import GameConfig
from gym_idsgame.envs.dao.idsgame_config import IdsGameConfig

class IDSEnvironment(IdsGameRandomAttackV21Env):
    """
    Extended environment for IDS analysis and training.
    Inherits from the most advanced random attack environment (v21) which includes:
    - 1 layer, 2 servers per layer
    - Dense rewards v3
    - Reconnaissance actions enabled
    - Partial observations
    - Random attacker starting position
    - Random environment
    """
    
    def __init__(self, save_dir: str = None, initial_state_path: str = None):
        """
        Initialize the v21 environment with default settings
        """
        super().__init__(idsgame_config=None, save_dir=save_dir, initial_state_path=initial_state_path)
        
    def analyze_state(self) -> Dict[str, Any]:
        """
        Analyze current state of the environment
        """
        return {
            "attack_values": self.state.attack_values.copy(),
            "defense_values": self.state.defense_values.copy(),
            "detection_values": self.state.defense_det.copy(),
            "reconnaissance_state": self.state.reconnaissance_state.copy(),
            "attacker_pos": self.state.attacker_pos,
            "game_step": self.state.game_step,
            "done": self.state.done,
            "detected": self.state.detected,
            "hacked": self.state.hacked,
            "reconnaissance_actions": self.state.reconnaissance_actions
        }
    
    def analyze_attack_defense_stats(self) -> Dict[str, Any]:
        """
        Analyze attack and defense statistics
        """
        return {
            "num_attacks": len(self.attacks),
            "num_failed_attacks": self.num_failed_attacks,
            "num_defenses": len(self.defenses),
            "num_detections": len(self.attack_detections),
            "hack_probability": self.hack_probability(),
            "attack_types": [attack[1] for attack in self.attacks],
            "defense_types": [defense[1] for defense in self.defenses],
            "reconnaissance_activities": len(self.state.reconnaissance_actions)
        }

    def run_episode(self, random_defense: bool = True) -> Dict[str, List[float]]:
        """
        Run a complete episode with optional random defense
        
        Args:
            random_defense: If True, use random defense actions
            
        Returns:
            Dictionary with episode statistics
        """
        obs, _ = self.reset()
        done = False
        episode_data = {
            "rewards": [],
            #"def_rewards": [],
            "attack_success": [],
            "detection_rate": [],
            "defense_effectiveness": [],
            "reconnaissance_rate": []
        }
        
        while not done:
            if random_defense:
                defense_action = self.defender_action_space.sample()
            else:
                # Minimal Defense Strategy
                defense_action = 0
                min_defense = float('inf')
                for d in range(self.num_defense_actions):
                    if self.is_defense_legal(d):
                        defense_val = self.state.defense_values.min()
                        if defense_val < min_defense:
                            min_defense = defense_val
                            defense_action = d
                            
            action = (-1, defense_action)  # -1 triggers random attack from attacker agent
            obs, reward, done, _, info = self.step(action)
            
            episode_data["rewards"].append(reward[0])  # Attacker reward
            #episode_data["def_rewards"].append(reward[1])  # Defender reward
            episode_data["attack_success"].append(1 if len(self.attacks) > 0 and 
                                                self.attacks[-1] not in self.failed_attacks else 0)
            episode_data["detection_rate"].append(len(self.attack_detections) / 
                                                max(1, len(self.attacks)))
            episode_data["defense_effectiveness"].append(self.num_failed_attacks / 
                                                       max(1, len(self.attacks)))
            episode_data["reconnaissance_rate"].append(len(self.state.reconnaissance_actions) /
                                                     max(1, len(self.attacks)))
            
        return episode_data
    
    def get_observation_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the observation space
        """
        attacker_obs, defender_obs = self.get_observation()
        
        return {
            "attacker_observation": {
                "shape": attacker_obs.shape,
                "range": (float(attacker_obs.min()), float(attacker_obs.max())),
                "reconnaissance_enabled": True,  # v21 always has reconnaissance
                "local_view": False  # v21 uses global view
            },
            "defender_observation": {
                "shape": defender_obs.shape,
                "range": (float(defender_obs.min()), float(defender_obs.max())),
                "fully_observed": self.fully_observed()
            },
            "network_config": {
                "num_layers": self.idsgame_config.game_config.num_layers,
                "num_servers_per_layer": self.idsgame_config.game_config.num_servers_per_layer,
                "num_attack_types": self.idsgame_config.game_config.num_attack_types,
                "max_value": self.idsgame_config.game_config.max_value
            }
        }
    
    def comprehensive_analysis(self) -> None:
        """
        Comprehensive analysis of the environment printing all available information
        """

        print("\n=== Environment Configuration ===")
        obs_info = self.get_observation_info()
        print("\nNetwork Configuration:")
        for k, v in obs_info["network_config"].items():
            print(f"- {k}: {v}")
            
        print("\nObservation Spaces:")
        print("\nAttacker Observation:")
        for k, v in obs_info["attacker_observation"].items():
            print(f"- {k}: {v}")
        print("\nDefender Observation:")
        for k, v in obs_info["defender_observation"].items():
            print(f"- {k}: {v}")
            
        print("\n=== Test Episode Analysis ===")
        episode_data = self.run_episode()
        
        print("\nEpisode Statistics:")
        print(f"- Total Steps: {len(episode_data['rewards'])}")
        print(f"- Average Reward: {np.mean(episode_data['rewards']):.2f}")
        #print(f"- Average Defender Reward: {np.mean(episode_data['def_rewards']):.2f}")
        print(f"- Attack Success Rate: {np.mean(episode_data['attack_success']):.2%}")
        print(f"- Detection Rate: {np.mean(episode_data['detection_rate']):.2%}")
        print(f"- Defense Effectiveness: {np.mean(episode_data['defense_effectiveness']):.2%}")
        print(f"- Reconnaissance Rate: {np.mean(episode_data['reconnaissance_rate']):.2%}")
        
        state_info = self.analyze_state()
        print("\nCurrent State Information:")
        print(f"- Attacker Position: {state_info['attacker_pos']}")
        print(f"- Game Step: {state_info['game_step']}")
        print(f"- Episode Done: {state_info['done']}")
        print(f"- Attacker Detected: {state_info['detected']}")
        print(f"- Target Hacked: {state_info['hacked']}")
        print(f"- Reconnaissance Actions: {len(state_info['reconnaissance_actions'])}")
        
        stats = self.analyze_attack_defense_stats()
        print("\nAttack-Defense Statistics:")
        print(f"- Total Attacks: {stats['num_attacks']}")
        print(f"- Failed Attacks: {stats['num_failed_attacks']}")
        print(f"- Total Defenses: {stats['num_defenses']}")
        print(f"- Attack Detections: {stats['num_detections']}")
        print(f"- Hack Probability: {stats['hack_probability']:.2%}")
        
        print("\nAction Spaces:")
        print(f"- Defender Actions: {self.num_defense_actions}")
        print(f"- Attack Actions: {self.num_attack_actions}")
        
        print("\nEnvironment Properties:")
        print(f"- Dense Rewards: {self.idsgame_config.game_config.dense_rewards}")
        print(f"- Reconnaissance Enabled: {self.idsgame_config.reconnaissance_actions}")
        print(f"- Random Starting Position: {self.idsgame_config.randomize_starting_position}")
        print(f"- Random Environment: {self.idsgame_config.randomize_env}")
        print(f"- Local View Observations: {self.idsgame_config.local_view_observations}")