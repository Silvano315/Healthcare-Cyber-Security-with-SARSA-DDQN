# Healthcare Cyber Security with SARSA & DDQN

## Table of Contents

- [Project Overview](#project-overview)
- [Environment](#environment)
- [Algorithms and Implementation](#algorithms-and-implementation)
- [SARSA](#sarsa)
  - [Results with Random Attack Strategy](#results-with-random-attack-strategy)
- [DDQN](#ddqn)
  - [Results with Random Attack Strategy](#results-with-random-attack-strategy)
  - [Results with Maximal Attack Strategy](#results-with-maximal-attack-strategy)
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

The project uses version 8 of the IDS game environment (`gym-idsgame-v8`) ([see documentation](https://github.com/Limmen/gym-idsgame/tree/master/experiments/training/v8)) for several reasons:
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

The implementation leverages `PyTorch`'s powerful deep learning capabilities, making the neural network architecture both flexible and efficient. PyTorch's intuitive API made it straightforward to implement both the online and target networks, with easy management of the backpropagation process and GPU acceleration when available.

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

### Results with Random Attack Strategy

The SARSA implementation was tested against a random attack strategy, producing several key insights (you can see [here](images/Sarsa_results_random_attack_seed_33.png) all the plots) based on both training and evaluation metrics:

#### Cumulative Rewards Analysis
- **Defender Performance**: Shows a strong positive trend in cumulative rewards, consistently increasing from around 0 to approximately 1250 over 20,000 episodes. This indicates that the defender progressively learned effective defense strategies.
- **Attacker Performance**: The attacker's cumulative rewards show a steady decline from 0 to around -800, suggesting that the defensive strategy effectively countered the random attacks over time.

#### Hack Probability
- Training shows a consistently low hack probability (around 0.2). Similar results have been obtained from Q-Learning example.

#### Exploration Rate (Epsilon)
The epsilon value shows a controlled linear decay from 1.0 to 0.2, indicating a proper balance between exploration and exploitation throughout the training process.

The overall results demonstrate that SARSA effectively learned a defensive strategy against random attacks.

## DDQN

### Results with Random Attack Strategy

The Double Deep Q-Network (DDQN) performance against random attacks shows clear learning patterns in the cumulative rewards (you can see [here](images/DDQN_results_random_attack_seed_33.png) all the plots):

#### Defender's Perspective
- **Training Phase**: The defender's cumulative reward demonstrates a strong linear growth pattern, rising from around 500 to 3000 units over the training episodes
- **Evaluation Phase**: Confirms the effective learning with a steady increase in cumulative rewards, reaching around 350 units by the end of evaluation

#### Attacker's Perspective
- **Training Phase**: The attacker's cumulative reward shows a steady decline from 0 to approximately -500 units
- **Evaluation Phase**: The decline is even more pronounced, dropping to around -300 units

These results demonstrate DDQN's ability to learn effective defensive policies, as evidenced by the inverse relationship between defender and attacker cumulative rewards - as the defender's rewards steadily increased, the attacker's correspondingly decreased, indicating successful defense against random attack strategies.

### Results with Maximal Attack Strategy

The DDQN's performance against maximal attacks shows interesting patterns in cumulative rewards (you can see [here](images/DDQN_results_maximal_attack_seed_33.png) all the plots), notably different from the random attack scenario:

#### Defender's Perspective
- **Training Phase**: Shows a distinctive learning curve, with an initial period of lower rewards followed by a strong positive trend reaching approximately 3500 units
- **Evaluation Phase**: Demonstrates consistent positive growth reaching around 1000 units, validating the robustness of the learned defensive strategy

#### Attacker's Perspective
- **Training Phase**: Shows an unusual pattern
  - Initial increase in cumulative rewards up to around 1000 units
  - Followed by a significant decline after episode 2000, eventually dropping to negative values
- **Evaluation Phase**: Shows a steady decline to around -1500 units

The results against maximal attacks demonstrate DDQN's ability to adapt and learn effective defensive strategies even against more sophisticated attack patterns. The initial spike in attacker rewards followed by a sharp decline suggests that the defender learned to counter initially successful attack strategies.

## 🎯 Key Insights

1. **Algorithm Performance Comparison**
   - SARSA demonstrated reliable performance against random attacks, with stable defensive learning
   - DDQN showed superior adaptability, performing well against both random and maximal attacks
   - Both algorithms successfully learned defensive strategies, as evidenced by consistently increasing defender rewards

2. **Attack Strategy Impact**
   - Random attacks were easier to defend against, showing consistent patterns in both algorithms
   - Maximal attacks initially posed a greater challenge (visible in DDQN's training curve), but were eventually countered effectively

3. **Learning Stability**
   - SARSA showed more stable learning curves with fewer fluctuations
   - DDQN exhibited more complex learning patterns but achieved higher cumulative rewards

4. **Implementation Insights**
   - The gym-idsgame environment, despite its age, proved to be a robust framework for testing security scenarios
   - The custom compatibility wrapper successfully bridged the gap between older and newer gym versions
   - PyTorch's implementation for DDQN provided efficient neural network management and training capabilities

## 🚀 How to Run

This project is designed to run smoothly on Google Colab. Follow these steps to get started:

### Setup Instructions

1. **Google Colab Configuration**
   ```bash
   # Clone the repository
   !git clone https://github.com/Silvano315/Healthcare-Cyber-Security-with-SARSA-DDQN.git
   
   # Navigate to project directory
   cd Healthcare-Cyber-Security-with-SARSA-DDQN
   
   # Copy missing gym utility files
   cp /content/Healthcare-Cyber-Security-with-SARSA-DDQN/missing_py_files_for_gym/atomic_write.py /content/Healthcare-Cyber-Security-with-SARSA-DDQN/missing_py_files_for_gym/closer.py /content/Healthcare-Cyber-Security-with-SARSA-DDQN/missing_py_files_for_gym/json_utils.py /usr/local/lib/python3.10/dist-packages/gym/utils/
   
   # Install required packages
   !pip install -r requirements.txt
   ```

2. **Post-Installation**
   - When pip installation completes, Colab will prompt you to restart the runtime
   - After restart, navigate back to the project directory:
   ```bash
   cd Healthcare-Cyber-Security-with-SARSA-DDQN
   ```

### Important Note
⚠️ When running on Google Colab, be aware that:
- The training process might appear to stall occasionally
- If this happens, wait a few minutes and refresh the page
- The training will continue correctly from where it left off
- Alternatively, you can reduce the number of episodes if experiencing persistent issues

The complete implementation and experimentation details can be found in the [notebook](./RL_project.ipynb)'s first section under **Google Colab Configuration**.