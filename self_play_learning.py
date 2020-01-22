from mcts import MCTS
import numpy as np
from tqdm import tqdm
from network import AlphaLoss, board_data
from copy import deepcopy
import pickle

class SelfPlay:

    def __init__(self, game, net, args):
        self.game = game
        self.net = net
        self.args = args
        self.mcts = MCTS(self.game, self.net, self.args)
        self.train_examples_history = [] # List of list of one iteration

    def run_episode(self):
        """ 
        Run one episode with player 1 starting.
        return : train_examples : list of (cannonical board, policy, value) value is +1 if the current player won the game else -1
        """
        train_examples = []
        board = self.game.init_state
        self.player_turn = 1
        episode_step = 0

        while True:
            episode_step += 1
            canonical_board = self.game.get_canonical_form(board, self.player_turn)
            temp = int(episode_step < self.args['tempThreshold'])
            pi = self.mcts.run_sims(canonical_board, temp=temp)
            train_examples.append([board, self.player_turn, pi, None])
            action = np.random.choice(len(pi), p=pi)
            board, self.player_turn = self.game.step(action, board, self.player_turn)

            winner = self.game.check_winner(board)

            if winner != 0:
                return [(x[0], x[2], 1 if (x[1] == winner) else -1 ) for x in train_examples]

    def learn(self):
        """ Performs num_iterations with n_epsisodes of self-play in each iteration"""

        for i in range(1, self.args['num_iterations']):
            print(f'Iteration {i}')
            train_examples_iter = []
            for episode in tqdm(range(self.args['num_episodes'])):
                self.mcts = MCTS(self.game, self.net, self.args)
                train_examples_iter += self.run_episode()
            self.train_examples_history.append(train_examples_iter)

        self.save_train_examples_history(i)

        if len(self.train_examples_history) > self.args['max_train_examples_history']:
            self.train_examples_history.pop(0)
        
        train_set = sum(self.train_examples_history, [])
        np.random.shuffle(train_set)
        train_set = board_data(train_set)
        
        # Save the untrained model 
        untrained_model = deepcopy(self.net)

        # Training the model

    
    def train_step(self, triplet):
        """ triplet is (state, target policy, target value) """
        state, target_pol, target_val = triplet

    def save_train_examples_history(self, iteration):
        pickle.save(self.train_examples_history, open(self.args['save_path'] + "iteration_" +str(iteration) + ".p", "wb"))



