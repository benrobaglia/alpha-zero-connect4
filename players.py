import numpy as np 

def random_play(board):
    valid_actions = np.where(board[0,:] == 0)[0]

    return np.random.choice(valid_actions)

