import numpy as np
from itertools import groupby, chain
import matplotlib.pyplot as plt 

class Connect4:
    def __init__(self, verbose=False):
        self.init_state = np.zeros([6,7]).astype(str)
        self.init_state[self.init_state == '0.0'] = " "
        self.turn = 0
        self.current_state = self.init_state
        self.name = 'connect4'
        self.done = False
        self.winner = " "
        self.verbose = verbose

    def step(self, action):
        player = self.get_player_turn()
        if self.current_state[0, action] != " ": # The column must not be full !
            print("Impossible action")
        else:
            for i in reversed(range(6)):
                if self.current_state[i, action] == " ":
                    pos = i
                    break
                
                self.current_state[pos, action] = self.players[player]

            self.winner = self.check_winner()
            if self.check_full() & (self.winner == ' '):
                self.winner =  'draw'
                self.done = True

            if self.winner != ' ':
                return self.winner
                self.done = True

            self.turn += 1

            if self.verbose:
                return self.current_state, self.players[self.get_player_turn()]
        
        
        
    def check_full(self):
        if ' ' not in self.current_state:
            return True
        else:
            return False
        
    def get_player_turn(self):
        return self.turn % 2
        
    def get_pos_diagonals (self, matrix):
        for di in ([(j, i - j) for j in range(6)] for i in range(6 + 7 -1)):
            yield [matrix[i][j] for i, j in di if i >= 0 and j >= 0 and i < 6 and j < 7]

    def get_neg_diagonals (self, matrix):
        for di in ([(j, i - 6 + j + 1) for j in range(6)] for i in range(6 + 7 - 1)):
            yield [matrix[i][j] for i, j in di if i >= 0 and j >= 0 and i < 6 and j < 7]

        
    def check_winner(self): 
        lines = (
            self.current_state, # columns
            zip(*self.current_state), # rows
            self.get_pos_diagonals(self.current_state), # positive diagonals
            self.get_neg_diagonals(self.current_state) # negative diagonals
        )

        for line in chain(*lines):
            for color, group in groupby(line):
                if color != ' ' and len(list(group)) >= 4:
                    return color
        return ' '

    
        
    def reset(self):
        self.init_state = np.zeros([6,7]).astype(str)
        self.init_state[self.init_state == '0.0'] = " "
    #             self.player = 0 # Tells us which turn it is to play
        self.turn = 0
        self.current_state = self.init_state

    def allowed_actions(self):
        acts = []
        for col in range(7):
            if self.current_state[0, col] == " ":
                acts.append(col)
        return acts

    def render(self):
        print(' |'.join(map(str, range(7))))
        for r in self.current_state:
            print(' |'.join(r))
    

    def to_canonical(self):
        if self.get_player_turn() == 0:
            pass
        else:
            self.turn += 1 # To fool get player turn 
            if self.winner == '0':
                self.winner = "X"
            elif self.winner == "X":
                self.winner = "0"
            self.current_state = np.where(self.current_state=="0", "00", self.current_state)
            self.current_state = np.where(self.current_state=="X", "0", self.current_state)
            self.current_state = np.where(self.current_state=="00", "0X", self.current_state)
