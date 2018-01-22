#!/usr/bin/env python3


class MultiAgentGrid(object):

    def __init__(self, gridSize=7):
        self.gridSize = gridSize
        self.defineMoves()

    def defineMoves(self):
        self.NOOP = 0
        self.LEFT = 1
        self.RIGHT = 2
        self.UP = 3
        self.DOWN = 4

    def reset_world(self):
        raise NotImplementedError
