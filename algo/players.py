import numpy as np 
from mcts import MCTS
import torch
from game import Connect4
from network import ConnectNet

args = {"exploration_constant":1.,
        "device":"cpu",
        "n_sim":25,
}
path = "networks_saved/net_init.pth"

game = Connect4()
net = ConnectNet()
net.load_state_dict(torch.load(path, map_location='cpu'))
mcts = MCTS(game, net, args)


def random_play(board):
    valid_actions = np.where(board[0,:] == 0)[0]

    return np.random.choice(valid_actions)

def human_player(board):
    valid_actions = np.where(board[0,:] == 0)[0]
    inp = int(input())
    if inp in valid_actions:
        return inp
    else:
        print("invalid action")


def alpha_zero_player(board):
    return np.argmax(mcts.run_sims(board, stochastic=False))