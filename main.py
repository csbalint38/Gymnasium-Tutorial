from gym.core import Env
import numpy as np
from gym import ActionWrapper, ObservationWrapper, RewardWrapper, Wrapper

import gymnasium as gym
from gymnasium.spaces import Box, Discrete

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
    
if __name__ == "__main__":
    env = gym.make("LunarLanderContinuous-v2")
    wrapped_env = DiscreteActions(env, [np.array[-1, 0], np.array[0, 1], np.array[0, -1]])
    print(wrapped_env.action_space)