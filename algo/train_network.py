import torch
import numpy as np
from game import Connect4
from mcts import MCTS
from network import ConnectNet, AlphaLoss
from self_play_learning import SelfPlay


args = {
        "alpha":0.8,
        "epsilon":0.2,
        "tauThreshold":10,
        "num_iterations":100,
        "num_episodes":100,
        "max_train_examples_history":20000,
        "batch_size":256,
        "n_challenges":30,
        "update_threshold":0.7,
        "save_path": "networks_saved/",
        "n_sim":50,
        "cpuct":1.,
        "device":"cpu",
        "n_epochs":10
        }

game = Connect4()
net = ConnectNet()

coach = SelfPlay(game, net, args)
coach.learn()
