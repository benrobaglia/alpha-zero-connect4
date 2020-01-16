import numpy as np
from itertools import groupby, chain
import matplotlib.pyplot as plt 
from copy import deepcopy

class Connect4:
    def __init__(self):
        self.init_state = np.zeros([6,7])
        self.name = 'connect4'

    def step(self, action, board, player_turn):
        new_board = deepcopy(board)
        if new_board[0, action] != 0.: # The column must not be full !
            return "Impossible action"
        else:
            for i in reversed(range(6)):
                if new_board[i, action] == 0.:
                    pos = i
                    break
            new_board[pos, action] = player_turn

        return new_board, - player_turn        
        
        
    def check_full(self, board):
        if 0 not in board:
            return True
        else:
            return False
        
        
    def get_pos_diagonals (self, matrix):
        for di in ([(j, i - j) for j in range(6)] for i in range(6 + 7 -1)):
            yield [matrix[i][j] for i, j in di if i >= 0 and j >= 0 and i < 6 and j < 7]

    def get_neg_diagonals (self, matrix):
        for di in ([(j, i - 6 + j + 1) for j in range(6)] for i in range(6 + 7 - 1)):
            yield [matrix[i][j] for i, j in di if i >= 0 and j >= 0 and i < 6 and j < 7]

        
    def check_winner(self, board): 
        lines = (
            board, # columns
            zip(*board), # rows
            self.get_pos_diagonals(board), # positive diagonals
            self.get_neg_diagonals(board) # negative diagonals
        )

        for line in chain(*lines):
            for color, group in groupby(line):
                if color != 0 and len(list(group)) >= 4:
                    return color
        
        if self.check_full(board):
            return 2 # It's a draw
        else:
            return 0 # Game has not ended

    def get_canonical_form(self, board, player):
        return board * player


    def allowed_actions(self, board):
        acts = []
        for col in range(7):
            if board[0, col] == 0:
                acts.append(col)
        return acts

    def render(self, board):
        print(' |'.join(map(str, range(7))))
        for r in board:
            print(' |'.join(str(r)))
    

