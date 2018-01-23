#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from gridEnvironment import MultiAgentGrid


class GridEnv(MultiAgentGrid):

    def __init__(self, gridSize=7, nAgents=2, seed=None):
        self.gridSize = gridSize
        self.numAgents = nAgents
        self.defineMoves()
        self.reset_world(nAgents, gridSize)

        if seed is not None:
            np.random.seed(seed)

    def reset_world(self):
        self.agents = []
        self.landmarks = []
        self.steps = 0

        for i in range(self.numAgents):
            self.agents.append(np.random.randint(low=0,
                                                 high=self.gridSize,
                                                 size=2))
        for i in range(self.numAgents):
            self.landmarks.append(np.random.randint(low=0,
                                                    high=self.gridSize,
                                                    size=2))

        self.agentReached = [False for i in range(self.numAgents)]

    def getState(self, agentNum, addId=True):
        """
        State is the relative positions of all other Landmarks and agents
        """

        relPositions = []
        for i in range(self.numAgents):
            if(i != agentNum):
                relPositions.append(self.agents[i] - self.agents[agentNum])
            relPositions.append(self.landmarks[i] - self.agents[agentNum])

        relPositions.append(self.agents[agentNum])

        if addId:
            relPositions.append([agentNum])

        return np.concatenate(relPositions)

    def act(self, actions, viz=False):
        """
        Returns Reward
        Reward of +1 first time designated landmark visited for agent
        Reward of all agets summed together
        """

        if len(actions) != self.numAgents:
            raise ValueError("Action size is incorrect")

        self.steps += 1
        reward = 0.0

        #  Negative reward proportional to closest landmark
        for i in range(self.numAgents):

            dist1 = np.min(np.sum(
                np.absolute(self.agents - self.landmarks[i]),
                axis=1))

            dist2 = np.min(
                np.sum(np.absolute(
                    np.absolute(self.agents - self.landmarks[i]) -
                    [self.gridSize, self.gridSize]),
                    axis=1))

            dist3 = np.min(
                np.sum(np.absolute(
                    np.absolute(self.agents - self.landmarks[i]) -
                    [self.gridSize, 0]),
                    axis=1))

            dist4 = np.min(
                np.sum(np.absolute(
                    np.absolute(self.agents - self.landmarks[i]) -
                    [0, self.gridSize]),
                    axis=1))

            dist = min(dist1, dist2, dist3, dist4)
            reward = reward - dist

            #  Penalize collisions
            for j in range(self.numAgents):
                if j != i and (np.sum(self.agents[i] == self.agents[j]) == 2):
                    reward -= self.gridSize

        # Update states of all agents
        for ind, act in enumerate(actions):
            if act == self.NOOP:
                pass
            elif act == self.LEFT:
                self.agents[ind] = (self.agents[ind] + [0, -1] +
                                    self.gridSize) % self.gridSize
            elif act == self.RIGHT:
                self.agents[ind] = (self.agents[ind] + [0, 1] +
                                    self.gridSize) % self.gridSize
            elif act == self.DOWN:
                self.agents[ind] = (self.agents[ind] + [1, 0] +
                                    self.gridSize) % self.gridSize
            elif act == self.UP:
                self.agents[ind] = (self.agents[ind] + [-1, 0] +
                                    self.gridSize) % self.gridSize
        if viz:
            self.visualizeState()

        return reward

    def visualizeState(self):
        if self.numAgents >= 6:
            raise ValueError("Visualization works for only if No." +
                             "of Agents < 6")

        N = self.gridSize
        # make an empty data set
        data = np.ones((N, N)) * np.nan

        for i in range(self.numAgents):
            x1, y1 = self.agents[i]
            x2, y2 = self.landmarks[i]
            data[x1][y1] = i + 1
            data[x2][y2] = 0

        colors = [
            '#566573',
            '#EC7063',
            '#AF7AC5',
            '#5499C7',
            '#1ABC9C',
            '#F4D03F']

        colors = colors[0: self.numAgents + 1]

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
        plt.savefig("img_cover/" + str(self.steps) + ".png")
        plt.close()
