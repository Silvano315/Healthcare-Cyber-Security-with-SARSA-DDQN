from typing import Union
import numpy as np
import time
import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from gym_idsgame.agents.training_agents.q_learning.q_agent import QAgent
from gym_idsgame.agents.training_agents.q_learning.q_agent_config import QAgentConfig
from gym_idsgame.envs.idsgame_env import IdsGameEnv
from gym_idsgame.envs.rendering.video.idsgame_monitor import IdsGameMonitor
from gym_idsgame.agents.dao.experiment_result import ExperimentResult
from gym_idsgame.agents.training_agents.models.fnn_w_linear import FNNwithLinear
from gym_idsgame.agents.training_agents.q_learning.experience_replay.replay_buffer import ReplayBuffer
from gym_idsgame.envs.constants import constants

class DDQNAgent(QAgent):
    """
    Double DQN implementation for the IDSGameEnv.
    Key difference from DQN: Uses two networks for action selection and evaluation
    to prevent overestimation of Q-values.
    """
    def __init__(self, env:IdsGameEnv, config: QAgentConfig):
        """
        Initialize environment and hyperparameters

        :param env: the environment to train on
        :param config: the hyperparameter configuration
        """
        super(DDQNAgent, self).__init__(env, config)

        self.attacker_q_network = None
        self.attacker_target_network = None
        self.defender_q_network = None
        self.defender_target_network = None
        self.loss_fn = None
        self.attacker_optimizer = None
        self.defender_optimizer = None
        self.attacker_lr_decay = None
        self.defender_lr_decay = None
        self.tensorboard_writer = SummaryWriter(self.config.dqn_config.tensorboard_dir)
        self.buffer = ReplayBuffer(config.dqn_config.replay_memory_size)
        self.initialize_models()
        self.tensorboard_writer.add_hparams(self.config.hparams_dict(), {})
        self.env.idsgame_config.save_trajectories = False
        self.env.idsgame_config.save_attack_stats = False

    def warmup(self) -> None:
        """
        A warmup without any learning just to populate the replay buffer following a random strategy

        :return: None
        """

        # Setup logging
        outer_warmup = tqdm.tqdm(total=self.config.dqn_config.replay_start_size, desc='Warmup', position=0)
        outer_warmup.set_description_str("[Warmup] step:{}, buffer_size: {}".format(0, 0))

        # Reset env
        obs = self.env.reset(update_stats=False)
        attacker_obs, defender_obs = obs
        obs_state_a = self.update_state(attacker_obs, defender_obs, attacker=True, state=[])
        obs_state_d = self.update_state(attacker_obs, defender_obs, attacker=False, state=[])
        obs = (obs_state_a, obs_state_d)
        self.config.logger.info("Starting warmup phase to fill replay buffer")

        # Perform <self.config.dqn_config.replay_start_size> steps and fill the replay memory
        for i in range(self.config.dqn_config.replay_start_size):

            if i % self.config.train_log_frequency == 0:
                log_str = "[Warmup] step:{}, buffer_size: {}".format(i, self.buffer.size())
                outer_warmup.set_description_str(log_str)
                self.config.logger.info(log_str)

            # Select random attacker and defender actions
            attacker_actions = list(range(self.env.num_attack_actions))
            defender_actions = list(range(self.env.num_defense_actions))
            legal_attack_actions = list(filter(lambda action: self.env.is_attack_legal(action), attacker_actions))
            legal_defense_actions = list(filter(lambda action: self.env.is_defense_legal(action), defender_actions))
            attacker_action = np.random.choice(legal_attack_actions)
            defender_action = np.random.choice(legal_defense_actions)
            action = (attacker_action, defender_action)

            # Take action in the environment
            obs_prime, reward, done, info = self.env.step(action)
            attacker_obs_prime, defender_obs_prime = obs_prime
            obs_state_a_prime = self.update_state(attacker_obs_prime, defender_obs_prime, attacker=True, state=[])
            obs_state_d_prime = self.update_state(attacker_obs_prime, defender_obs_prime, attacker=False, state=[])
            obs_prime = (obs_state_a_prime, obs_state_d_prime)

            # Add transition to replay memory
            self.buffer.add_tuple(obs, action, reward, done, obs_prime)

            #print(f"In warmup after add tuple: {obs, action, reward, done, obs_prime}")

            # Move to new state
            obs = obs_prime
            outer_warmup.update(1)

            if done:
                obs = self.env.reset(update_stats=False)
                attacker_obs, defender_obs = obs
                obs_state_a = self.update_state(attacker_obs, defender_obs, attacker=True, state=[])
                obs_state_d = self.update_state(attacker_obs, defender_obs, attacker=False, state=[])
                obs = (obs_state_a, obs_state_d)

        self.config.logger.info("{} Warmup steps completed, replay buffer size: {}".format(
            self.config.dqn_config.replay_start_size, self.buffer.size()))
        self.env.close()

        try:
            # Add network graph to tensorboard with a sample batch as input
            mini_batch = self.buffer.sample(self.config.dqn_config.batch_size)
            s_attacker_batch, s_defender_batch, a_attacker_batch, a_defender_batch, r_attacker_batch, r_defender_batch, \
            d_batch, s2_attacker_batch, s2_defender_batch = mini_batch

            if self.config.attacker:
                s_1 = torch.tensor(s_attacker_batch).float()
                # Move to GPU if using GPU
                if torch.cuda.is_available() and self.config.dqn_config.gpu:
                    device = torch.device("cuda:0")
                    s_1 = s_1.to(device)

                self.tensorboard_writer.add_graph(self.attacker_q_network, s_1)

            if self.config.defender:

                s_1 = torch.tensor(s_defender_batch).float()
                # Move to GPU if using GPU
                if torch.cuda.is_available() and self.config.dqn_config.gpu:
                    device = torch.device("cuda:0")
                    s_1 = s_1.to(device)

                self.tensorboard_writer.add_graph(self.defender_q_network, s_1)
        except:
            self.config.logger.warning("Error when trying to add network graph to tensorboard")

    def initialize_models(self) -> None:
        """
        Initialize models
        :return: None
        """

        # Initialize models
        self.attacker_q_network = FNNwithLinear(self.config.dqn_config.input_dim, self.config.dqn_config.attacker_output_dim,
                                                self.config.dqn_config.hidden_dim,
                                                num_hidden_layers=self.config.dqn_config.num_hidden_layers,
                                                hidden_activation=self.config.dqn_config.hidden_activation)
        self.attacker_target_network = FNNwithLinear(self.config.dqn_config.input_dim, self.config.dqn_config.attacker_output_dim,
                                                     self.config.dqn_config.hidden_dim,
                                                     num_hidden_layers=self.config.dqn_config.num_hidden_layers,
                                                     hidden_activation=self.config.dqn_config.hidden_activation)
        self.defender_q_network = FNNwithLinear(self.config.dqn_config.input_dim, self.config.dqn_config.defender_output_dim,
                                                self.config.dqn_config.hidden_dim,
                                                num_hidden_layers=self.config.dqn_config.num_hidden_layers,
                                                hidden_activation=self.config.dqn_config.hidden_activation)
        self.defender_target_network = FNNwithLinear(self.config.dqn_config.input_dim, self.config.dqn_config.defender_output_dim,
                                                     self.config.dqn_config.hidden_dim,
                                                     num_hidden_layers=self.config.dqn_config.num_hidden_layers,
                                                     hidden_activation=self.config.dqn_config.hidden_activation)

        # Specify device
        if torch.cuda.is_available() and self.config.dqn_config.gpu:
            device = torch.device("cuda:0")
            self.config.logger.info("Running on the GPU")
        else:
            device = torch.device("cpu")
            self.config.logger.info("Running on the CPU")

        self.attacker_q_network.to(device)
        self.attacker_target_network.to(device)
        self.defender_q_network.to(device)
        self.defender_target_network.to(device)

        # Set the target network to use the same weights initialization as the q-network
        self.attacker_target_network.load_state_dict(self.attacker_q_network.state_dict())
        self.defender_target_network.load_state_dict(self.defender_q_network.state_dict())
        # The target network is not trainable it is only used for predictions, therefore we set it to eval mode
        # to turn of dropouts, batch norms, gradient computations etc.
        self.attacker_target_network.eval()
        self.defender_target_network.eval()

        # Construct loss function
        if self.config.dqn_config.loss_fn == "MSE":
            self.loss_fn = torch.nn.MSELoss()
        elif self.config.dqn_config.loss_fn == "Huber":
            self.loss_fn = torch.nn.SmoothL1Loss()
        else:
            raise ValueError("Loss function not recognized")

        # Define Optimizer. The call to model.parameters() in the optimizer constructor will contain the learnable
        # parameters of the layers in the model
        if self.config.dqn_config.optimizer == "Adam":
            self.attacker_optimizer = torch.optim.Adam(self.attacker_q_network.parameters(), lr=self.config.alpha)
            self.defender_optimizer = torch.optim.Adam(self.defender_q_network.parameters(), lr=self.config.alpha)
        elif self.config.dqn_config.optimizer == "SGD":
            self.attacker_optimizer = torch.optim.SGD(self.attacker_q_network.parameters(), lr=self.config.alpha)
            self.defender_optimizer = torch.optim.SGD(self.defender_q_network.parameters(), lr=self.config.alpha)
        else:
            raise ValueError("Optimizer not recognized")

        # LR decay
        if self.config.dqn_config.lr_exp_decay:
            self.attacker_lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.attacker_optimizer,
                                                                       gamma=self.config.dqn_config.lr_decay_rate)
            self.defender_lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.attacker_optimizer,
                                                                            gamma=self.config.dqn_config.lr_decay_rate)
            
    def training_step(self, mini_batch: Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                      np.ndarray, np.ndarray, np.ndarray], attacker: bool = True) -> torch.Tensor:
        """
        Performs a training step using Double DQN algorithm.
        Key differences from DQN:
        1. Action selection using online network
        2. Value evaluation using target network
        
        :param mini_batch: a minibatch of transitions
        :param attacker: whether doing a training step for attacker or defender
        :return: loss
        """
        # Unpack batch
        s_attacker_batch, s_defender_batch, a_attacker_batch, a_defender_batch, r_attacker_batch, r_defender_batch, \
        d_batch, s2_attacker_batch, s2_defender_batch = mini_batch

        # Convert to tensors and set network modes
        if attacker:
            self.attacker_q_network.train()
            self.attacker_target_network.eval()
            r_1 = torch.tensor(r_attacker_batch).float()
            s_1 = torch.tensor(s_attacker_batch).float()
            s_2 = torch.tensor(s2_attacker_batch).float()
        else:
            self.defender_q_network.train()
            self.defender_target_network.eval()
            r_1 = torch.tensor(r_defender_batch).float()
            s_1 = torch.tensor(s_defender_batch).float()
            s_2 = torch.tensor(s2_defender_batch).float()

        # Move to GPU if available
        if torch.cuda.is_available() and self.config.dqn_config.gpu:
            device = torch.device("cuda:0")
            r_1 = r_1.to(device)
            s_1 = s_1.to(device)
            s_2 = s_2.to(device)

        # Initialize target baseline
        if attacker:
            target = self.attacker_q_network(s_1)
        else:
            target = self.defender_q_network(s_1)

        # DDQN Update !!!! Here you can find the most important difference with DQN agent
        with torch.no_grad():
            if attacker:
                # Select actions using online network
                next_actions = torch.argmax(self.attacker_q_network(s_2), dim=1)
                # Evaluate actions using target network
                next_q_values = self.attacker_target_network(s_2)
            else:
                # Select actions using online network
                next_actions = torch.argmax(self.defender_q_network(s_2), dim=1)
                # Evaluate actions using target network
                next_q_values = self.defender_target_network(s_2)
            
            # Gather Q-values for selected actions
            next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze()

        # Update targets
        for i in range(self.config.dqn_config.batch_size):
            if d_batch[i]:  # Terminal state
                if attacker:
                    target[i][a_attacker_batch[i]] = r_1[i]
                else:
                    target[i][a_defender_batch[i]] = r_1[i]
            else:  # Non-terminal state
                if attacker:
                    target[i][a_attacker_batch[i]] = r_1[i] + self.config.gamma * next_q_values[i]
                else:
                    target[i][a_defender_batch[i]] = r_1[i] + self.config.gamma * next_q_values[i]

        # Compute prediction and loss
        if attacker:
            prediction = self.attacker_q_network(s_1)
        else:
            prediction = self.defender_q_network(s_1)
        
        loss = self.loss_fn(prediction, target)

        # Optimization step
        if attacker:
            self.attacker_optimizer.zero_grad()
            loss.backward()
            self.attacker_optimizer.step()
        else:
            self.defender_optimizer.zero_grad()
            loss.backward()
            self.defender_optimizer.step()

        return loss
    
    def get_action(self, state: np.ndarray, eval: bool = False, attacker: bool = True) -> int:
        """
        Gets an action using an epsilon-greedy policy with respect to the Q-network.
        For DDQN, we always use the online network for action selection.

        :param state: the state to get an action for
        :param eval: whether in evaluation mode (affects epsilon)
        :param attacker: whether getting action for attacker or defender
        :return: The selected action
        """
        if not attacker and (not isinstance(state, np.ndarray) or state.dtype == object):
            state = self.env.state.get_defender_observation(
                self.env.idsgame_config.game_config.network_config)
        state = torch.from_numpy(state.flatten()).float()

        # Move to GPU if available
        if torch.cuda.is_available() and self.config.dqn_config.gpu:
            device = torch.device("cuda:0")
            state = state.to(device)

        # Get legal actions
        if attacker:
            actions = list(range(self.env.num_attack_actions))
            legal_actions = list(filter(lambda action: self.env.is_attack_legal(action), actions))
        else:
            actions = list(range(self.env.num_defense_actions))
            legal_actions = list(filter(lambda action: self.env.is_defense_legal(action), actions))

        # Epsilon-greedy action selection
        if (np.random.rand() < self.config.epsilon and not eval) \
                or (eval and np.random.random() < self.config.eval_epsilon):
            return np.random.choice(legal_actions)

        # Get Q-values using online network
        with torch.no_grad():
            if attacker:
                act_values = self.attacker_q_network(state)
            else:
                act_values = self.defender_q_network(state)

        # Select best legal action
        return legal_actions[torch.argmax(act_values[legal_actions]).item()]

    def train(self) -> ExperimentResult:
        """
        Trains the DDQN agent. Similar to DQN but with different update mechanism.

        :return: ExperimentResult
        """
        self.config.logger.info("Starting Training")
        if len(self.train_result.avg_episode_steps) > 0:
            self.config.logger.warning("starting training with non-empty result object")
        
        # Initial warmup to fill replay buffer
        self.warmup()
        
        # Training setup
        done = False
        obs = self.env.reset(update_stats=False)
        attacker_obs, defender_obs = obs
        obs_state_a = self.update_state(attacker_obs, defender_obs, attacker=True, state=[])
        obs_state_d = self.update_state(attacker_obs, defender_obs, attacker=False, state=[])
        obs = (obs_state_a, obs_state_d)

        # Training metrics
        episode_attacker_rewards = []
        episode_defender_rewards = []
        episode_steps = []
        episode_avg_attacker_loss = []
        episode_avg_defender_loss = []

        # Logging
        self.outer_train = tqdm.tqdm(total=self.config.num_episodes, desc='Train Episode', position=0)
        self.outer_train.set_description_str("[Train] epsilon:{:.2f},avg_a_R:{:.2f},avg_d_R:{:.2f},"
                                             "avg_t:{:.2f},avg_h:{:.2f},acc_A_R:{:.2f}," \
                                             "acc_D_R:{:.2f}".format(self.config.epsilon, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        # Main training loop
        for episode in range(self.config.num_episodes):
            episode_attacker_reward = 0
            episode_defender_reward = 0
            episode_step = 0
            episode_attacker_loss = 0.0
            episode_defender_loss = 0.0
            
            while not done:
                if self.config.render:
                    self.env.render(mode="human")

                # Get actions
                attacker_action = 0
                defender_action = 0
                if self.config.attacker:
                    attacker_action = self.get_action(obs[0], attacker=True)
                if self.config.defender:
                    defender_action = self.get_action(obs[1], attacker=False)
                action = (attacker_action, defender_action)

                # Take step in environment
                obs_prime, reward, done, _ = self.env.step(action)
                
                # Process new state
                attacker_obs_prime, defender_obs_prime = obs_prime
                obs_state_a_prime = self.update_state(attacker_obs_prime, defender_obs_prime, attacker=True, state=[])
                obs_state_d_prime = self.update_state(attacker_obs_prime, defender_obs_prime, attacker=False, state=[])
                obs_prime = (obs_state_a_prime, obs_state_d_prime)

                # Store transition
                self.buffer.add_tuple(obs, action, reward, done, obs_prime)

                # DDQN update if buffer has enough samples
                if self.buffer.size() >= self.config.dqn_config.batch_size:
                    minibatch = self.buffer.sample(self.config.dqn_config.batch_size)
                    
                    if self.config.attacker:
                        loss = self.training_step(minibatch, attacker=True)
                        episode_attacker_loss += loss.item()
                    
                    if self.config.defender:
                        loss = self.training_step(minibatch, attacker=False)
                        episode_defender_loss += loss.item()

                # Update metrics
                attacker_reward, defender_reward = reward
                episode_attacker_reward += attacker_reward
                episode_defender_reward += defender_reward
                episode_step += 1
                obs = obs_prime

            # End of episode processing
            if self.config.render:
                self.env.render(mode="human")

            # Update target network
            if episode % self.config.dqn_config.target_network_update_freq == 0:
                self.update_target_network()

            # Decay learning rate if configured
            if self.config.dqn_config.lr_exp_decay:
                if self.config.attacker:
                    self.attacker_lr_decay.step()
                if self.config.defender:
                    self.defender_lr_decay.step()

            # Record episode metrics
            self.num_train_games += 1
            self.num_train_games_total += 1
            if self.env.state.hacked:
                self.num_train_hacks += 1
                self.num_train_hacks_total += 1
                
            episode_attacker_rewards.append(episode_attacker_reward)
            episode_defender_rewards.append(episode_defender_reward)
            
            if episode_step > 0:
                if self.config.attacker:
                    episode_avg_attacker_loss.append(episode_attacker_loss/episode_step)
                if self.config.defender:
                    episode_avg_defender_loss.append(episode_defender_loss/episode_step)
            else:
                if self.config.attacker:
                    episode_avg_attacker_loss.append(episode_attacker_loss)
                if self.config.defender:
                    episode_avg_defender_loss.append(episode_defender_loss)
            
            episode_steps.append(episode_step)

            # Logging
            if episode % self.config.train_log_frequency == 0:
                self.log_metrics(episode, self.train_result, episode_attacker_rewards,
                               episode_defender_rewards, episode_steps, 
                               episode_avg_attacker_loss, episode_avg_defender_loss)
                
                episode_attacker_rewards = []
                episode_defender_rewards = []
                episode_steps = []

            # Evaluation
            if episode % self.config.eval_frequency == 0:
                self.eval(episode)

            # Save models
            if episode % self.config.checkpoint_freq == 0:
                self.save_model()
                if self.config.save_dir is not None:
                    time_str = str(time.time())
                    self.train_result.to_csv(self.config.save_dir + "/" + time_str + "_train_results_checkpoint.csv")
                    self.eval_result.to_csv(self.config.save_dir + "/" + time_str + "_eval_results_checkpoint.csv")

            # Reset for next episode
            done = False
            obs = self.env.reset(update_stats=True)
            attacker_obs, defender_obs = obs
            obs_state_a = self.update_state(attacker_obs, defender_obs, attacker=True, state=[])
            obs_state_d = self.update_state(attacker_obs, defender_obs, attacker=False, state=[])
            obs = (obs_state_a, obs_state_d)

            # Anneal epsilon
            self.anneal_epsilon()

        # Training complete
        self.config.logger.info("Training Complete")
        
        # Final evaluation
        self.eval(self.config.num_episodes-1, log=False)
        
        # Save final model and data
        self.save_model()

        if self.config.save_dir is not None:
            time_str = str(time.time())
            self.train_result.to_csv(self.config.save_dir + "/" + time_str + "_train_results_checkpoint.csv")
            self.eval_result.to_csv(self.config.save_dir + "/" + time_str + "_eval_results_checkpoint.csv")
        
        return self.train_result

    def update_target_network(self) -> None:
        """
        Updates the target networks with the current Q-networks' weights
        
        :return: None
        """
        self.config.logger.info("Updating target network")

        if self.config.attacker:
            self.attacker_target_network.load_state_dict(self.attacker_q_network.state_dict())
            self.attacker_target_network.eval()

        if self.config.defender:
            self.defender_target_network.load_state_dict(self.defender_q_network.state_dict())
            self.defender_target_network.eval()

    def eval(self, train_episode, log=True) -> ExperimentResult:
        """
        Performs evaluation with the greedy policy with respect to the learned Q-values

        :param train_episode: the train episode to keep track of logging
        :param log: whether to log the result
        :return: None
        """
        self.config.logger.info("Starting Evaluation")
        time_str = str(time.time())

        self.num_eval_games = 0
        self.num_eval_hacks = 0

        if len(self.eval_result.avg_episode_steps) > 0:
            self.config.logger.warning("starting eval with non-empty result object")
        if self.config.eval_episodes < 1:
            return
        done = False

        # Video config
        if self.config.video:
            if self.config.video_dir is None:
                raise AssertionError("Video is set to True but no video_dir is provided, please specify "
                                     "the video_dir argument")
            self.env = IdsGameMonitor(self.env, self.config.video_dir + "/" + time_str, force=True,
                                      video_frequency=self.config.video_frequency)
            self.env.metadata["video.frames_per_second"] = self.config.video_fps

        # Tracking metrics
        episode_attacker_rewards = []
        episode_defender_rewards = []
        episode_steps = []

        # Logging
        self.outer_eval = tqdm.tqdm(total=self.config.eval_episodes, desc='Eval Episode', position=1)
        self.outer_eval.set_description_str(
            "[Eval] avg_a_R:{:.2f},avg_d_R:{:.2f},avg_t:{:.2f},avg_h:{:.2f},acc_A_R:{:.2f}," \
            "acc_D_R:{:.2f}".format(0.0, 0,0, 0.0, 0.0, 0.0, 0.0))

        # Eval
        obs = self.env.reset(update_stats=False)
        attacker_obs, defender_obs = obs
        obs_state_a = self.update_state(attacker_obs, defender_obs, attacker=True, state=[])
        obs_state_d = self.update_state(attacker_obs, defender_obs, attacker=False, state=[])
        obs = (obs_state_a, obs_state_d)
        attacker_obs, defender_obs = obs

        for episode in range(self.config.eval_episodes):
            episode_attacker_reward = 0
            episode_defender_reward = 0
            episode_step = 0
            while not done:
                if self.config.eval_render:
                    self.env.render()
                    time.sleep(self.config.eval_sleep)

                # Default initialization
                attacker_action = 0
                defender_action = 0

                # Get attacker and defender actions
                if self.config.attacker:
                    attacker_action = self.get_action(attacker_obs, eval=True, attacker=True)
                if self.config.defender:
                    defender_action = self.get_action(defender_obs, eval=True, attacker=False)
                action = (attacker_action, defender_action)

                # Take a step in the environment
                obs_prime, reward, done, _ = self.env.step(action)
                attacker_obs_prime, defender_obs_prime = obs_prime
                obs_state_a_prime = self.update_state(attacker_obs_prime, defender_obs_prime, attacker=True, state=[])
                obs_state_d_prime = self.update_state(attacker_obs_prime, defender_obs_prime, attacker=False, state=[])
                obs_prime = (obs_state_a_prime, obs_state_d_prime)

                # Update state information and metrics
                attacker_reward, defender_reward = reward
                obs_prime_attacker, obs_prime_defender = obs_prime
                episode_attacker_reward += attacker_reward
                episode_defender_reward += defender_reward
                episode_step += 1
                attacker_obs = obs_prime_attacker
                defender_obs = obs_prime_defender

            # Render final frame when game completed
            if self.config.eval_render:
                self.env.render()
                time.sleep(self.config.eval_sleep)
            self.config.logger.info("Eval episode: {}, Game ended after {} steps".format(episode, episode_step))

            # Record episode metrics
            episode_attacker_rewards.append(episode_attacker_reward)
            episode_defender_rewards.append(episode_defender_reward)
            episode_steps.append(episode_step)

            # Update eval stats
            self.num_eval_games +=1
            self.num_eval_games_total += 1
            if self.env.state.detected:
                self.eval_attacker_cumulative_reward -= constants.GAME_CONFIG.POSITIVE_REWARD
                self.eval_defender_cumulative_reward += constants.GAME_CONFIG.POSITIVE_REWARD
            if self.env.state.hacked:
                self.eval_attacker_cumulative_reward += constants.GAME_CONFIG.POSITIVE_REWARD
                self.eval_defender_cumulative_reward -= constants.GAME_CONFIG.POSITIVE_REWARD
                self.num_eval_hacks += 1
                self.num_eval_hacks_total +=1

            # Log average metrics every <self.config.eval_log_frequency> episodes
            if episode % self.config.eval_log_frequency == 0 and log:
                if self.num_eval_hacks > 0:
                    self.eval_hack_probability = float(self.num_eval_hacks) / float(self.num_eval_games)
                if self.num_eval_games_total > 0:
                    self.eval_cumulative_hack_probability = float(self.num_eval_hacks_total) / float(
                        self.num_eval_games_total)
                self.log_metrics(episode, self.eval_result, episode_attacker_rewards, episode_defender_rewards, episode_steps,
                                 eval = True, update_stats=False)

            # Save gifs
            if self.config.gifs and self.config.video:
                self.env.generate_gif(self.config.gif_dir + "/episode_" + str(train_episode) + "_"
                                      + time_str + ".gif", self.config.video_fps)

                # Add frames to tensorboard
                for idx, frame in enumerate(self.env.episode_frames):
                    self.tensorboard_writer.add_image(str(train_episode) + "_eval_frames/" + str(idx),
                                                       frame, global_step=train_episode,
                                                      dataformats = "HWC")


            # Reset for new eval episode
            done = False
            obs = self.env.reset(update_stats=False)
            attacker_obs, defender_obs = obs
            obs_state_a = self.update_state(attacker_obs, defender_obs, attacker=True, state=[])
            obs_state_d = self.update_state(attacker_obs, defender_obs, attacker=False, state=[])
            obs = (obs_state_a, obs_state_d)
            attacker_obs, defender_obs = obs
            self.outer_eval.update(1)

        # Log average eval statistics
        if log:
            if self.num_eval_hacks > 0:
                self.eval_hack_probability = float(self.num_eval_hacks) / float(self.num_eval_games)
            if self.num_eval_games_total > 0:
                self.eval_cumulative_hack_probability = float(self.num_eval_hacks_total) / float(
                    self.num_eval_games_total)

            self.log_metrics(train_episode, self.eval_result, episode_attacker_rewards, episode_defender_rewards,
                             episode_steps, eval=True, update_stats=True)

        self.env.close()
        self.config.logger.info("Evaluation Complete")
        return self.eval_result
    
    def save_model(self) -> None:
        """
        Saves the neural network models to disk
        
        :return: None
        """
        time_str = str(time.time())
        if self.config.save_dir is not None:
            if self.config.attacker:
                path = self.config.save_dir + "/" + time_str + "_attacker_ddqn_network.pt"
                self.config.logger.info("Saving Q-network to: {}".format(path))
                torch.save(self.attacker_q_network.state_dict(), path)
            if self.config.defender:
                path = self.config.save_dir + "/" + time_str + "_defender_ddqn_network.pt"
                self.config.logger.info("Saving Q-network to: {}".format(path))
                torch.save(self.defender_q_network.state_dict(), path)
        else:
            self.config.logger.warning("Save path not defined, not saving Q-networks to disk")

    def update_state(self, attacker_obs: np.ndarray = None, defender_obs: np.ndarray = None,
                 state: np.ndarray = None, attacker: bool = True) -> np.ndarray:
        """
        Update approximative Markov state. Processes raw observations to create the state
        representation used by the networks.

        :param attacker_obs: attacker observation
        :param defender_obs: defender observation
        :param state: current state
        :param attacker: boolean flag whether it is attacker or not
        :return: processed state representation
        """
        if attacker and self.env.idsgame_config.game_config.reconnaissance_actions:
            a_obs_len = self.env.idsgame_config.game_config.num_attack_types + 1
            defender_obs = attacker_obs[:, a_obs_len:a_obs_len+self.env.idsgame_config.game_config.num_attack_types]
            if self.env.idsgame_config.reconnaissance_bool_features:
                d_bool_features = attacker_obs[:, a_obs_len+self.env.idsgame_config.game_config.num_attack_types:]
            attacker_obs = attacker_obs[:, 0:a_obs_len]

        """if not attacker and self.env.local_view_features():
            attacker_obs = self.env.state.get_attacker_observation(
                self.env.idsgame_config.game_config.network_config,
                local_view=False,
                reconnaissance=self.env.idsgame_config.reconnaissance_actions)
            defender_obs = self.env.state.get_defender_observation(
                self.env.idsgame_config.game_config.network_config)"""

        if not attacker and self.env.local_view_features():
            attacker_obs = self.env.state.get_attacker_observation(
                self.env.idsgame_config.game_config.network_config,
                local_view=False,
                reconnaissance=self.env.idsgame_config.reconnaissance_actions)

        # Feature normalization
        if self.config.dqn_config.normalize_features:
            # Normalize attacker observations
            if not self.env.local_view_features() or not attacker:
                attacker_obs_1 = attacker_obs[:, 0:-1] / np.linalg.norm(attacker_obs[:, 0:-1])
            else:
                attacker_obs_1 = attacker_obs[:, 0:-2] / np.linalg.norm(attacker_obs[:, 0:-2])
            
            normalized_attacker_features = []
            for idx, row in enumerate(attacker_obs_1):
                if np.isnan(attacker_obs_1).any():
                    t = attacker_obs[idx]
                else:
                    t = row.tolist()
                    if not self.env.local_view_features() or not attacker:
                        t.append(attacker_obs[idx][-1])
                    else:
                        t.append(attacker_obs[idx][-2])
                        t.append(attacker_obs[idx][-1])
                normalized_attacker_features.append(t)
            attacker_obs = np.array(normalized_attacker_features)

            # Normalize defender observations similarly
            if attacker and self.env.idsgame_config.game_config.reconnaissance_actions:
                defender_obs_1 = defender_obs[:, 0:-1] / np.linalg.norm(defender_obs[:, 0:-1])
            else:
                defender_obs_1 = defender_obs / np.linalg.norm(defender_obs)
            
            normalized_defender_features = []
            for idx, row in enumerate(defender_obs_1):
                if np.isnan(defender_obs_1).any():
                    t = defender_obs[idx]
                else:
                    if attacker and self.env.idsgame_config.game_config.reconnaissance_actions:
                        t = row.tolist()
                        t.append(defender_obs[idx][-1])
                    else:
                        t = row
                normalized_defender_features.append(t)
            defender_obs = np.array(normalized_defender_features)

        # State composition based on view type
        if self.env.local_view_features() and attacker:
            if not self.env.idsgame_config.game_config.reconnaissance_actions:
                neighbor_defense_attributes = np.zeros((attacker_obs.shape[0], defender_obs.shape[1]))
                for node in range(attacker_obs.shape[0]):
                    id = int(attacker_obs[node][-1])
                    neighbor_defense_attributes[node] = defender_obs[id]
            else:
                neighbor_defense_attributes = defender_obs

        # Final state assembly
        if self.env.fully_observed() or \
                (self.env.idsgame_config.game_config.reconnaissance_actions and attacker):
            if self.config.dqn_config.merged_ad_features:
                # Merge attacker and defender features
                if not self.env.local_view_features() or not attacker:
                    features = self._merge_global_features(attacker_obs, defender_obs)
                else:
                    features = self._merge_local_features(attacker_obs, neighbor_defense_attributes)
                
                if self.env.idsgame_config.reconnaissance_bool_features:
                    features = self._add_bool_features(features, d_bool_features)
                
                return self._process_state_length(features, state)
            else:
                return self._process_partial_observability(attacker_obs, defender_obs, 
                                                        neighbor_defense_attributes, state, attacker)
        else:
            if self.config.dqn_config.state_length == 1:
                return np.array(attacker_obs if attacker else defender_obs)
            if len(state) == 0:
                return np.array([attacker_obs if attacker else defender_obs] * self.config.dqn_config.state_length)
            
            state = np.append(state[1:], 
                            np.array([attacker_obs if attacker else defender_obs]), 
                            axis=0)
            return state