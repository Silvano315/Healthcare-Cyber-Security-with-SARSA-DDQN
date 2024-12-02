import numpy as np
import gymnasium as gym
from typing import Union

class GymCompatibilityWrapper(gym.Wrapper):
    """
    A wrapper to make gym-idsgame environment compatible with latest Gymnasium API.

    This wrapper handles API differences between gym-idsgame and Gymnasium by:
    - Managing update_stats parameter in reset
    - Providing a compatible render method for video recording
    - Adapting the step method return format

    Attributes:
        env: The wrapped gym-idsgame environment
    """

    def reset(self, **kwargs) -> tuple:
        """
        Reset the environment while handling the update_stats parameter.
        
        Args:
            **kwargs: Keyword arguments for reset, including update_stats
                
        Returns:
            tuple: (attacker_obs, defender_obs)
        """
        update_stats = kwargs.get('update_stats', False)
        self.env.reset(update_stats=update_stats)
        return self.env.get_observation()

    # TO BE REFACTORED (if you want to see gifs or video)
    def render(self, *args, **kwargs) -> Union[None, np.ndarray]:
        """
        Render the environment state.

        Provides compatibility for video recording by returning a dummy frame
        when rgb_array mode is requested.

        Args:
            *args: Positional arguments for render
            **kwargs: Keyword arguments for render, including mode
            
        Returns:
            Union[None, np.ndarray]: Either None for human rendering or 
                                    a dummy frame for rgb_array mode
        """
        if 'mode' in kwargs and kwargs['mode'] == 'rgb_array':
            return [np.zeros((64, 64, 3), dtype=np.uint8)]
        return self.env.render()

    def step(self, action: tuple) -> tuple:
        """
        Execute one environment step.
        
        Adapts the step method to maintain compatibility between 
        gym-idsgame and Gymnasium APIs.
        
        Args:
            action: tuple of (attacker_action, defender_action)
            
        Returns:
            tuple: (observation, reward, done, info) where:
                    - observation is the full environment observation
                    - reward is (attacker_reward, defender_reward)
                    - done is whether the episode has ended
                    - info is a dict of additional information
        """
        full_obs = self.env.get_observation()
        _, reward, terminated, truncated, info = self.env.step(action)
        return full_obs, reward, terminated or truncated, info