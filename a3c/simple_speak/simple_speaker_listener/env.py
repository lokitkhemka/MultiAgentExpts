#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from gridEnvironment import MultiAgentGrid


class GridEnv(MultiAgentGrid):

    def __init__(self, gridSize=7, landMarks=3, seed=None):
        self.gridSize = gridSize
        self.landMarksNum = landMarks
        self.defineMoves()
        if seed is not None:
            np.random.seed(seed)

        self.reset_world()

    def reset_world(self):
        self.landmarks = []
        self.steps = 0

        self.agent = np.random.randint(low=0,
                                       high=self.gridSize,
                                       size=2)

        self.target = np.random.randint(low=0,
                                        high=self.landMarksNum)

        for i in range(self.landMarksNum):
            self.landmarks.append(np.random.randint(low=0,
                                                    high=self.gridSize,
                                                    size=2))

    def getState(self, agentNum, addId=True):
        """
        State is the relative positions of all other Landmarks and agents
        """

        if agentNum == 0:
            relPositions = []
            for i in range(self.landMarksNum):
                if(i != agentNum):
                    relPositions.append(self.landmarks[i] - self.agent)
            relPositions.append(self.agent)
            return np.concatenate(relPositions)

        else:
            return np.asarray([self.target])

    def act(self, action, viz=False):
        """
        Returns Reward
        Reward of +1 first time designated landmark visited for agent
        Reward of all agets summed together
        """

        self.steps += 1
        reward = 0.0

        x1, y1 = self.agent
        x2, y2 = self.landmarks[self.target]

        xdist = min(abs(x1 - x2), self.gridSize - abs(x1 - x2))
        ydist = min(abs(y1 - y2), self.gridSize - abs(y1 - y2))
        reward = reward - xdist - ydist

        reward = reward / float(self.gridSize)

        #  Update states of all agents
        if action == self.NOOP:
            pass
        elif action == self.LEFT:
            self.agent = (self.agent + [0, -1] +
                          self.gridSize) % self.gridSize
        elif action == self.RIGHT:
            self.agent = (self.agent + [0, 1] +
                          self.gridSize) % self.gridSize
        elif action == self.DOWN:
            self.agent = (self.agent + [1, 0] +
                          self.gridSize) % self.gridSize
        elif action == self.UP:
            self.agent = (self.agent + [-1, 0] +
                          self.gridSize) % self.gridSize
        if viz:
            self.visualizeState()

        return reward

    def visualizeState(self):
        if self.landMarksNum >= 7:
            raise ValueError("Visualization works for only if No." +
                             "of Agents < 7")

        N = self.gridSize
        # make an empty data set
        data = np.ones((N, N)) * np.nan

        x1, y1 = self.agent

        for i in range(self.landMarksNum):
            x2, y2 = self.landmarks[i]
            data[x2][y2] = i
            if i == self.target:
                data[x1][y1] = i

        colors = [
            '#E74C3C',
            '#9B59B6',
            '#2980B9',
            '#17A589',
            '#F1C40F',
            '#566573']

        colors = colors[0: self.landMarksNum]

        # make a figure + axes
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        # make color map
        my_cmap = matplotlib.colors.ListedColormap(colors)
        # set the 'bad' values (nan) to be white and transparent
        my_cmap.set_bad(color='w', alpha=0)
        # draw the grid
        for x in range(N + 1):
            ax.axhline(x, lw=2, color='k', zorder=5)
            ax.axvline(x, lw=2, color='k', zorder=5)
        # draw the boxes
        ax.imshow(data, interpolation='none', cmap=my_cmap,
                  extent=[0, N, 0, N], zorder=0)
        # turn off the axis labels
        ax.axis('off')
        plt.savefig("img_speak/" + str(self.steps) + ".png")
        plt.close()
