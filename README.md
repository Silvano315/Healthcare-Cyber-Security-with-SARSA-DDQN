# Healthcare Cyber Security with SARSA & DDQN

## Table of Contents

- [Project Overview](#project-overview)
- [Environment](#environment)
- [Algorithms and Implementation](#algorithms-and-implementation)
- [SARSA](#sarsa)
  - [Results](#results)
- [DDQN](#ddqn)
  - [Results](#results)
- [Key Insights](#key-insights)
- [How to Run](#how-to-run)


## 🌟 Project Overview

This repository is the eighth project of the master's degree in AI Engineering with [Profession AI](https://profession.ai), all the credits for the requests and idea go to this team. 

DeepGuard Inc., a leading healthcare cybersecurity company, is facing an increase in the complexity and sophistication of cyber attacks. The need to protect sensitive patient information and ensure regulatory compliance is critical to maintaining customer trust and data security.

Project Benefits:
- Defense Improvement
- Vulnerability Identification
- Resource Optimization

The main goal is to implement advanced Reinforcement Learning algorithms to simulate and mitigate attack and defense scenarios within the [gym-idsgame](https://github.com/Limmen/gym-idsgame) environment, specialized in attack and defense simulations in computer networks. I need to use techniques such as:
- **SARSA Algorithm**: to address "random attack" scenarios in the gym-idsgame environment.
- **DDQN Algorithm with PyTorch**: to solve "random attack" and "maximal attack" scenarios.

## 🍽️ Environment

This project utilizes the [gym-idsgame](https://github.com/Limmen/gym-idsgame) environment, specifically version 21 ([see documentation](https://github.com/Limmen/gym-idsgame/tree/master/experiments/training/v21)), which is designed for training and evaluating reinforcement learning agents in network security scenarios.

The environment supports different attack strategies:
- **Random Attack**: The attacker randomly selects actions from the action space
- **Maximal Attack**: The attacker always chooses the action that maximizes the immediate reward

For a deeper understanding of the environment's capabilities and features, you can explore the `IDSGameExplorer` class implementation in [`src.environment.explorer`](src/environment/explorer.py). This custom class, showcased in the [RL_project.ipynb](./RL_project.ipynb) notebook under the "IDS-Game Environment Exploration" section with global and specific analysis, provides tools to:
- Visualize the environment state
- Understand possible actions and rewards
- Explore state transitions
- Analyze and visualize defense and attack patterns
- Test different attack and defense scenarios
- Visualize render environment 
- Analyse network structure

## 🛠️ Algorithms and Implementation

### Project Challenges

The main challenge in this project wasn't just implementing SARSA and DDQN algorithms, but rather managing and adapting an outdated environment to work with modern frameworks. Several critical issues had to be addressed:

1. **Environment Compatibility**: The `gym-idsgame` library was several years old and incompatible with the current Gymnasium API. Key differences included:
   - Different return values from the `step()` method
   - Inconsistent API signatures
   - Outdated dependencies

2. **Custom Wrapper Development**: To bridge these compatibility gaps, I developed a custom wrapper [here](src/environment/compatibility_wrapper.py).

3. **Missing Dependencies**: When running on Google Colab, several utility files were missing and had to be manually sourced and integrated.

### Environment Selection

The project uses version 8 of the IDS game environment (`gym-idsgame-v8`) for several reasons:
- Comprehensive documentation of random attack scenarios
- Existing baseline performance metrics for various algorithms
- Well-structured for comparative analysis

### Algorithm Implementation

#### SARSA (State-Action-Reward-State-Action)
- Built upon the existing Q-learning implementation in the `gym-idsgame` library
- Key characteristics:
  - On-policy learning algorithm
  - Uses actual next action for updates
  - More conservative learning compared to Q-learning

#### DDQN (Double Deep Q-Network)
- Extended from the existing DQN implementation
- Key improvements over DQN:
  - Uses two networks (online and target) to reduce overestimation
  - More stable learning through decoupled action selection and evaluation

### Implementation Approach

I maintained consistency with the `gym-idsgame` library's architecture because it offers:
1. Robust logging and monitoring systems
2. Built-in visualization tools
3. Efficient configuration management
4. Ready-to-use evaluation metrics

For detailed configuration options and hyperparameter tuning (such as γ, α, neural network architectures), please refer to the [RL_project.ipynb](./RL_project.ipynb) notebook. The notebook provides an interactive environment where you can:
- Modify learning rates and discount factors
- Adjust neural network architectures for DDQN
- Experiment with different environment parameters
- Visualize and analyze training results

## SARSA

## DDQN

## 🎯 Key Insights

## 🚀 How to Run
