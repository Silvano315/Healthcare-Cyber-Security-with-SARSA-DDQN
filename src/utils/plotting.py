import os
import matplotlib.pyplot as plt
from matplotlib.image import imread


def plot_results(output_dir: str, random_seed: int, algorithm: str, mode: str) -> None:
    """
    Create a comparison figure showing training and evaluation results from saved plots.

    This function creates a 7x2 grid of plots comparing training and evaluation metrics
    for a reinforcement learning experiment. It loads pre-generated plot images from
    the specified directory and arranges them in a clear comparative layout.

    Args:
        output_dir: Base directory containing the results
        random_seed: Random seed used for the experiment
        algorithm: Name of the algorithm used (e.g., "SARSA", "DDQN")
        mode: name of the env mode (e.g., 'random', 'maximal')
        
    Training plots:
        - defender_cumulative_reward_train.png
        - hack_probability_train.png
        - attacker_cumulative_reward_train.png
        - avg_episode_lengths_train.png
        - avg_attacker_episode_returns_train.png
        - avg_defender_episode_returns_train.png
        - epsilon_train.png
    And corresponding evaluation plots with '_eval' suffix.

    Returns:
        None: Displays the composite figure using matplotlib
    """
    plot_dir = os.path.join(output_dir, 'results/plots', str(random_seed))
    
    training_plots = [
        "defender_cumulative_reward_train.png",
        "hack_probability_train.png",
        "attacker_cumulative_reward_train.png",
        "avg_episode_lengths_train.png",
        "avg_attacker_episode_returns_train.png",
        "avg_defender_episode_returns_train.png",
        "epsilon_train.png"
    ]
    evaluation_plots = [
        name.replace("_train", "_eval") for name in training_plots
    ]
    
    figure_title = algorithm + f" Results with {mode.capitalize()} Attack"
    left_column_title = "Training Results"
    right_column_title = "Evaluation Results"
    fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(14, 28))
    fig.suptitle(figure_title, fontsize=25, y=0.92)
    axes = axes.flatten()
    
    for i, file_name in enumerate(training_plots):
        file_path = os.path.join(plot_dir, file_name)
        if os.path.exists(file_path):
            img = imread(file_path)
            axes[2*i].imshow(img)
            axes[2*i].axis('off')  
        axes[2*i].set_title(file_name.replace("_train.png", "").replace("_", " ").capitalize(), fontsize=18)
    
    for i, file_name in enumerate(evaluation_plots):
        file_path = os.path.join(plot_dir, file_name)
        if os.path.exists(file_path):
            img = imread(file_path)
            axes[2*i+1].imshow(img)
            axes[2*i+1].axis('off') 
        axes[2*i+1].set_title(file_name.replace("_eval.png", "").replace("_", " ").capitalize(), fontsize=18)
    
    fig.text(0.25, 0.90, left_column_title, ha='center', fontsize=20)
    fig.text(0.75, 0.90, right_column_title, ha='center', fontsize=20)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, wspace=0.4)
    plt.show()