import torch
import numpy as np
from game import Connect4
from mcts import MCTS
from network import ConnectNet, AlphaLoss
from arena import Arena
from self_play_learning import SelfPlay


args = {"stochasticThreshold":15,
        "num_iterations":10000,
        "num_episodes":500,
        "max_train_examples_history":200000,
        "batch_size":256,
        "n_challenges":50,
        "update_threshold":0.7,
        "save_path": "networks_saved/",
        "n_sim":25,
        "exploration_constant":1.,
        "device":"cuda",
        "n_epochs":10}

game = Connect4()
net = ConnectNet()

coach = SelfPlay(game, net, args)
coach.learn()
