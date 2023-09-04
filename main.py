from gym.core import Env
import numpy as np
from gym import ActionWrapper, ObservationWrapper, RewardWrapper, Wrapper

import gymnasium as gym
from gymnasium.spaces import Box, Discrete

from typing import SupportsFloat

class RelativePosition(ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self._observation_space = Box(shape=(2,), low=np.inf, high=np.inf)

    def observation(self, observation):
        return observation['target'] - observation['agent']
    
class DiscreteActions(ActionWrapper):
    def __init__(self, env: Env, disc_to_cont):
        super().__init__(env)
        self.disc_to_cont = disc_to_cont
        self.action_space = Discrete(len(disc_to_cont))

    def action(self, act):
        return self.disc_to_cont[act]
    
class ClipReward(RewardWrapper):
    def __init__(self, env: Env, min_reward, max_reward):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (min_reward, max_reward)

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        return np.clip(reward, self.min_reward, self.max_reward)
    
class ReacherRewardWrapper(Wrapper):
    def __init__(self, env: Env, reward_dist_weight, reward_ctrl_weight):
        super().__init__(env)
        self.reward_dist_weight = reward_dist_weight
        self.reward_ctrl_weight = reward_ctrl_weight

    def step(self, action):
        obs, _, terminated, trunctated, info = self.env.step(action)
        reward = (self.reward_dist_weight * info["reward_dist"], self.reward_ctrl_weight * info["reward_ctrl"])
        return obs, reward, terminated, trunctated, info
    
if __name__ == "__main__":
    env = gym.make("LunarLanderContinuous-v2")
    wrapped_env = DiscreteActions(env, [np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])])
    print(wrapped_env.action_space)