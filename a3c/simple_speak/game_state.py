# -*- coding: utf-8 -*-
import numpy as np
from constants import GAME

if GAME == 0:
    from simple_speaker_listener.env import GridEnv as env


class GameState(object):
    def __init__(self, rand_seed):
        self.env = env(gridSize=8, landMarks=3, seed=rand_seed)
        self.reset()

    def reset(self):
        # randomize initial state
        self.env.reset_world()

        st = self.env.getState(0, addId=False)
        self.s_t = np.concatenate([st, np.zeros((5))])
        s2Tmp = np.zeros([3])
        s2Tmp[self.env.getState(1, addId=False)] = 1
        self.s2 = s2Tmp

    def process(self, action, comm):
        # convert original 18 action index to minimal action set index
        self.reward = self.env.act(action)
        st = self.env.getState(0, addId=False)
        commApp = np.zeros((5))
        commApp[comm] = 1
        self.s_t1 = np.concatenate([st, commApp])

    def update(self):
        self.s_t = self.s_t1
