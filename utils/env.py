import gym
import gym_minigrid
from gym_minigrid.wrappers import *


def make_env(env_key, seed=None):
    env = gym.make(env_key)
    # env = FullyObsWrapper(env) # Get full grid (25 x 25 rather than 7 x 7)
    # env = RGBImgPartialObsWrapper(env)  # Get pixel observations
    # env = ImgObsWrapper(env)  # Get rid of the 'mission' field
    # env = ActionBonus(env)
    # env = StateBonus(env)
    env.seed(seed)
    return env
