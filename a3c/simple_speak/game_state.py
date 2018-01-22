# -*- coding: utf-8 -*-
import numpy as np
from constants import GAME

if GAME == 0:
    from MultiAgentDomains.src.simple_speaker_listener.env import GridEnv as env


class GameState(object):
    def __init__(self, rand_seed):

        self.env = env(gridSize=8, landmarks=3, seed=rand_seed)
        self.reset()

    def reset(self):
        self.ale.reset_game()

        # randomize initial state
        self.env.reset_world()

        st = self.env.getState(0, addId=False)
        self.s_t = np.concatenate([st, np.zeros((5))])

    def process(self, action, comm):
        # convert original 18 action index to minimal action set index
        self.reward = self.env.act(action)
        st = self.env.getState(0, addId=False)
        self.s_t1 = np.concatenate([st, comm])

    def update(self):
        self.s_t = self.s_t1
