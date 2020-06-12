from mcts import MCTS
import numpy as np
from tqdm import tqdm
from network import AlphaLoss, board_data
import pickle
from torch.utils.data import TensorDataset, DataLoader
import torch

class SelfPlay:

    def __init__(self, game, net, args):
        self.game = game
        self.args = args
        self.net = net.to(self.args['device'])
        self.not_trained_net = net.__class__(net.n_blocks).to(self.args['device'])
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        self.train_examples_history = [] # List of list of one iteration
        self.history_loss = []
        self.criterion = AlphaLoss()

    def run_episode(self):
        """ 
        Run one episode with player 1 starting.
        return : train_examples : list of (cannonical board, policy, value) value is +1 if the current player won the game else -1
        """
        train_examples = []
        board = self.game.init_state
        self.player_turn = 1
        episode_step = 0
        mcts = MCTS(board, self.player_turn, self.game, self.net, self.args)

        while True:
            episode_step += 1
            tau = float(episode_step < self.args['tauThreshold'])
            pi, _ = mcts.get_action_probs(tau=tau)
            train_examples.append([board, self.player_turn, pi, None])
            action = np.random.choice(len(pi), p=pi)
            board, self.player_turn = self.game.step(action, board, self.player_turn)
            mcts.set_root(board)

            winner = self.game.check_winner(board)

            if winner != 0:
                return [(x[0], x[2], 1 if (x[1] == winner) else -1 ) for x in train_examples]

    def learn(self):
        """ Performs num_iterations with n_epsisodes of self-play in each iteration"""

        for i in range(1, self.args['num_iterations']):
            print(f'Iteration {i}')
            train_examples_iter = []
            print("Getting experience...")
            for _ in tqdm(range(self.args['num_episodes'])):
                train_examples_iter += self.run_episode()
            self.train_examples_history.append(train_examples_iter)

            self.save_train_examples_history(i)

            if len(self.train_examples_history) > self.args['max_train_examples_history']:
                self.train_examples_history.pop(0)
            
            if len(self.train_examples_history[0]) > 2:
                print("Start training...")
                # Prepare data for training
                train_set = sum(self.train_examples_history, [])
                
                np.random.shuffle(train_set)
                states, target_policies, target_values = zip(*train_set)

                states = np.stack(states)
                target_policies = np.array(target_policies)
                target_values = np.array(target_values)

                dataset = TensorDataset(
                    torch.tensor(states, requires_grad=True).float(),
                    torch.tensor(target_policies, requires_grad=False).float(),
                    torch.tensor(target_values,  requires_grad=False).float()
                    )
                dataloader = DataLoader(dataset, batch_size=self.args['batch_size'])

                # Set the untrained model and its MCTS
                self.not_trained_net.load_state_dict(self.net.state_dict())
                
                # Training the model
                for _ in range(self.args['n_epochs']):
                    loss_epoch = []
                    for sample in dataloader:
                        loss = self.train_step(sample)
                        loss_epoch.append(loss)
                    print(f'loss epoch: {np.mean(loss_epoch)}')

                # Set mcts of the trained model
                trained_mcts = MCTS(self.game.init_state, 1, self.game, self.net, self.args)
                not_trained_mcts = MCTS(self.game.init_state, 1, self.game, self.not_trained_net, self.args)


                print('Start evaluation...')
                stats = {"trained_wins":0, "draws":0, "not_trained_wins":0}

                # Trained starts
                for _ in tqdm(range(self.args['n_challenges']//2)):
                    outcome = self.play_game(trained_mcts, not_trained_mcts)
                    if outcome == 1:
                        stats['trained_wins'] += 1
                    elif outcome == 2:
                        stats['draws'] += 1
                    else:
                        stats['not_trained_wins'] += 1

                # Not trained starts
                for _ in tqdm(range(self.args['n_challenges']//2)):
                    outcome = self.play_game(not_trained_mcts, trained_mcts)
                    if outcome == -1:
                        stats['trained_wins'] += 1
                    elif outcome == 2:
                        stats['draws'] += 1
                    else:
                        stats['not_trained_wins'] += 1

                # Accepting or rejecting the model
                if stats['not_trained_wins'] + stats['trained_wins'] != 0:
                    p_accept = stats['trained_wins'] / (stats['not_trained_wins'] + stats['trained_wins'])
                else:
                    p_accept = 0
                print("pourcentage trained wins : ", p_accept)

                print(f"Not trained net wins {stats['not_trained_wins']} times, Trained net wins {stats['trained_wins']}, {stats['draws']} draws.")
                if p_accept > self.args['update_threshold']:
                    torch.save(self.net.state_dict(), f"{self.args['save_path']}best_net_iteration_{i}.pth")
                    print("Model accepted !")
                # else:
                    self.net.load_state_dict(self.not_trained_net.state_dict())


    
    def train_step(self, sample):
        """ triplet is (state, target policy, target value) """
        state, policy, value = sample
        state = state.to(self.args['device'])
        policy = policy.to(self.args['device'])
        value = value.to(self.args['device'])

        policy_pred, value_pred = self.net(state)
        loss = self.criterion(value_pred, value, policy_pred, policy)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def play_game(self, mcts1, mcts2, verbose=False):
        board = self.game.init_state
        player_turn = 1
        episode_step = 0
        winner = 0

        while winner == 0:
            if verbose:
                print("Episode step:", episode_step)
                print(board)

            episode_step += 1
            mcts1.set_root(board)
            mcts2.set_root(board)
            
            if player_turn == 1:
                pi,_ = mcts1.get_action_probs(tau=0.01) 
            else:    
                pi, _ = mcts2.get_action_probs(tau=0.01) 
            
            if len(pi) > 7:
                print(pi)

            action = np.argmax(pi)
            board, player_turn = self.game.step(action, board, player_turn)

            winner = self.game.check_winner(board)
        

        return winner


    def save_train_examples_history(self, iteration):
        pickle.dump(self.train_examples_history, open(self.args['save_path'] + "train_samples_iteration.p", "wb"))



