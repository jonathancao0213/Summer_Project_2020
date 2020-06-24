import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class TradingEnv(gym.Env):
    def __init__(self, data):
        super(Agent, self).__init__()

        self.data = data
        self.current_day = 0

        #initial balance of 10,000
        self.balance = 10,000

        #number of stock multipled by the opening price of that day
        self.number_of_stocks = 0

        #Three actions: buy, sell, or nothing
        self.action_space = space.Discrete(3)

        #change shape to the data
        self.observation_space = space.Box(low = 0, high = 1, shape = (6,6), dtype = np.float16)

    def _next_day(self):
        next_day_data = data[self.current_day+1,:]
        np.append(next_day_data, self.balance)
        return next_day_data

    def _take_action(self, action):
        #if do nothing
        if action == 0:
            pass
        #if buy stock
        elif action == 1:
            #subtract balance and by the day close and
            self.balance -= data[self.current_day, 1]
            self.number_of_stocks += 1
        #if sell stock
        elif action == 2:
            self.balance += data[self.current_day, 1]*self.number_of_stocks
            self.number_of_stocks = 0


    def step(self, action):
        self._take_action(action)

        self.current_day += 1

        reward = (self.balance + self.number_of_stocks*data[self.current_day,1])  * (self.current_day/365)

        done = (self.balance + self.number_of_stocks*data[self.current_day,1]) <= 0

        observations = self._next_day()

        return observations, reward, done, {}

    def reset(self):
        self.balance = 10,000
        self.number_of_stocks = 0
        self.current_day = 0

    return self._next_day()
