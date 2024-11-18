import numpy as np
import time
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import matplotlib.pyplot as plt
from src.utils.logger import setup_logger

class DefenseTrainer:
    """
    Trainer class for defensive agents in the IdsGame environment.
    Handles training loop, evaluation, metrics tracking, and visualization.
    """
    
    def __init__(self, agent: Any, env_wrapper: Any, config: Dict[str, Any]):
        """
        Initialize the trainer.
        
        Args:
            agent: The defensive agent (SARSA or DDQN)
            env_wrapper: Wrapped IdsGame environment
            config: Configuration dictionary from config.yaml
        """
        self.agent = agent
        self.env = env_wrapper
        self.config = config
        
        self.logger = setup_logger(
            name="DefenseTrainer",
            level=config['logging']['level'],
            log_dir=config['logging']['dir'],
            filename=config['logging']['filename']
        )
        
        self.num_episodes = config['sarsa']['training']['num_episodes']
        self.max_steps = config['sarsa']['training']['max_steps_per_episode']
        self.eval_frequency = config['sarsa']['training']['evaluation']['frequency']
        self.eval_episodes = config['sarsa']['training']['evaluation']['episodes']
        
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'defense_rates': [],
            'evaluation_scores': []
        }
        
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories for saving models and results."""
        os.makedirs(self.config['sarsa']['checkpoint']['dir'], exist_ok=True)
        os.makedirs(self.config['visualization']['plots_dir'], exist_ok=True)
        os.makedirs(self.config['logging']['dir'], exist_ok=True)
        
    def train(self) -> Dict[str, List[float]]:
        """
        Train the defensive agent.
        
        Returns:
            Dictionary containing training metrics
        """
        self.logger.info("Starting defensive training...")
        start_time = time.time()
        
        for episode in range(self.num_episodes):
            episode_reward = 0
            state, _ = self.env.reset()
            
            for step in range(self.max_steps):
                action = self.agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                metrics = self.agent.train_step(state, action, reward, next_state, 
                                              terminated or truncated)
                
                episode_reward += reward
                state = next_state
                
                if terminated or truncated:
                    break
            
            self.training_metrics['episode_rewards'].append(episode_reward)
            self.training_metrics['episode_lengths'].append(step + 1)
            self.training_metrics['defense_rates'].append(self.agent.get_defense_rate())
            
            if episode % self.config['sarsa']['logging']['frequency'] == 0:
                self.log_training_progress(episode, metrics)
            
            if episode % self.eval_frequency == 0:
                eval_score = self.evaluate()
                self.training_metrics['evaluation_scores'].append(eval_score)
                self.logger.info(f"Evaluation at episode {episode}: {eval_score:.2f}")
            
            if episode % self.config['sarsa']['checkpoint']['frequency'] == 0:
                self.save_checkpoint(episode)
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        self.save_results()
        self.create_training_plots()
        
        return self.training_metrics
    
    def evaluate(self) -> float:
        """
        Evaluate the current defense policy.
        
        Returns:
            Average defense success rate during evaluation
        """
        eval_rewards = []
        eval_defense_rates = []
        
        for _ in range(self.eval_episodes):
            episode_reward = 0
            state, _ = self.env.reset()
            defense_successes = 0
            total_attempts = 0
            
            for _ in range(self.max_steps):
                action = self.agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                episode_reward += reward
                total_attempts += 1
                if reward > 0:  
                    defense_successes += 1
                    
                state = next_state
                if terminated or truncated:
                    break
            
            eval_rewards.append(episode_reward)
            defense_rate = defense_successes / total_attempts if total_attempts > 0 else 0
            eval_defense_rates.append(defense_rate)
        
        return np.mean(eval_defense_rates)
    
    def log_training_progress(self, episode: int, metrics: Dict[str, float]):
        """
        Log training progress.
        
        Args:
            episode: Current episode number
            metrics: Dictionary of current metrics
        """
        avg_reward = np.mean(self.training_metrics['episode_rewards'][-100:])
        avg_defense_rate = np.mean(self.training_metrics['defense_rates'][-100:])
        
        self.logger.info(
            f"Episode {episode}/{self.num_episodes} - "
            f"Avg Reward: {avg_reward:.2f}, "
            f"Defense Rate: {avg_defense_rate:.2%}, "
            f"Epsilon: {metrics.get('epsilon', 0):.3f}"
        )
    
    def save_checkpoint(self, episode: int):
        """
        Save a training checkpoint.
        
        Args:
            episode: Current episode number
        """
        checkpoint_path = os.path.join(
            self.config['sarsa']['checkpoint']['dir'],
            f"defense_agent_episode_{episode}.pt"
        )
        self.agent.save(checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def save_results(self):
        """Save training results and metrics."""
        results = {
            'training_metrics': self.training_metrics,
            'config': self.config,
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        }
        
        results_path = os.path.join(
            self.config['logging']['dir'],
            'training_results.json'
        )
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
    
    def create_training_plots(self):
        """Create and save visualization plots."""
        if not self.config['visualization']['show_training_progress']:
            return
            
        plt.figure(figsize=(15, 10))
        
        # Plot episode rewards
        plt.subplot(2, 2, 1)
        plt.plot(self.training_metrics['episode_rewards'])
        plt.title('Defense Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # Plot defense success rate
        plt.subplot(2, 2, 2)
        plt.plot(self.training_metrics['defense_rates'])
        plt.title('Defense Success Rate')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        
        # Plot evaluation scores
        plt.subplot(2, 2, 3)
        plt.plot(range(0, self.num_episodes, self.eval_frequency), 
                self.training_metrics['evaluation_scores'])
        plt.title('Evaluation Defense Rate')
        plt.xlabel('Episode')
        plt.ylabel('Defense Rate')
        
        # Plot episode lengths
        plt.subplot(2, 2, 4)
        plt.plot(self.training_metrics['episode_lengths'])
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.config['visualization']['plots_dir'],
            'training_results.png'
        ))
        plt.close()