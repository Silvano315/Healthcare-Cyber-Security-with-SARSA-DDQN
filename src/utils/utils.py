import os
import numpy as np
from datetime import datetime
from gym_idsgame.agents.dao.experiment_result import ExperimentResult

def get_output_dir(algorithm: str) -> str:
    """Create and get output directory

        Args:
        algorithm: Name of the algorithm used (e.g., "sarsa", "ddqn")
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"results/training/{algorithm}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def print_summary(result: ExperimentResult, title: str) -> None:
    """
    Print detailed summary statistics of training or evaluation results

    Args:
        result: ExperimentResult object containing training/evaluation metrics and history
        title: Title string to display at the top of the summary (e.g. "Training" or "Evaluation")
    """
    print(f"\n" + title + " Summary:")
    print("-" * 50)
    
    rewards = result.avg_defender_episode_rewards
    steps = result.avg_episode_steps
    hack_prob = result.hack_probability
    
    print(f"Final Defense Performance:")
    print(f"- Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"- Max-Min Reward: {np.max(rewards)} - {np.min(rewards)}")
    print(f"- Average Episode Length: {np.mean(steps):.2f} ± {np.std(steps):.2f}")
    print(f"- Max-Min Episode Length: {np.max(steps)} - {np.min(steps)}")
    print(f"- Average Hack Probability: {np.mean(hack_prob):.2%} ± {np.std(hack_prob):.2%}")
    print(f"- Max-Min Hack Probability: {np.max(hack_prob)} - {np.min(hack_prob)}")
    print(f"- Final Cumulative Reward: {result.defender_cumulative_reward[-1]}")