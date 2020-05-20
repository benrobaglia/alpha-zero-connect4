import numpy as np
from game import Connect4
from arena import Arena
from players import random_play

game = Connect4()
jeu = Arena(random_play, random_play, game)

jeu.play_game(verbose=True)
