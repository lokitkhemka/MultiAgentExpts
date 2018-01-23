# -*- coding: utf-8 -*-
from constants import GAME

if GAME == 1:
    from simple_cover.env import GridEnv as env


class GameState(object):
    def __init__(self, rand_seed):
        self.env = env(gridSize=7, nAgents=2, seed=rand_seed)
        self.reset()

    def reset(self):
        # randomize initial state
        self.env.reset_world()

        self.s1_t = self.env.getState(0, addId=True)
        self.s2_t = self.env.getState(1, addId=True)

    def process(self, action):
        # convert original 18 action index to minimal action set index
        self.reward = self.env.act(action)
        self.s1_tx = self.env.getState(0, addId=True)
        self.s2_tx = self.env.getState(1, addId=True)

    def update(self):
        self.s1_t = self.s1_tx
        self.s2_t = self.s1_tx
