import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import envs
import pandas as pd

data = pd.read_csv('../../../Data/NVDA_raw_data.csv')

headers = data.columns.tolist()
print(data.values)
env = gym.make('TradingEnv-v0', data = data.values)
print(env.observation_space)
close_index = headers.index('Close')
