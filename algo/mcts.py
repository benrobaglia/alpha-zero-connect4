import numpy as np
import torch

epsilon=1e-7

class MCTS:
    """
    Monte Carlo Tree search algorithm
    """

    def __init__(self, game, net, args):
        self.game = game
        self.net = net
        self.args = args
        self.Qsa = {}       # Q values for state s and action a
        self.Nsa = {}       # number of times s,a was visited
        self.Ns = {}        # number of times board s was visited
        self.Ps = {}        # initial policy

        self.Es = {}        # game status for board s (winner, not ended, etc.)
        self.Vs = {}        # valid actions for board s

    def run_sims(self, board, stochastic=1.):
        """
        Board : cannonical board : 6x7 matrix. Player 1 to play
        Stochastic: bool = 1 if we have a stochastic policy
        """
        for _ in range(self.args['n_sim']):
            self.search(board) # Run n_sim searches

        s = str(board)
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(7)]

        if stochastic == 0.:
            best_action = np.argmax(counts)
            policy = [0]*len(counts)
            policy[best_action] = 1
            return policy

        else:
            counts = [x for x in counts]
            counts_sum = np.sum(counts)
            policy = [x/counts_sum for x in counts]
            return policy

    def search(self, board):
        s = str(board)

        if s not in self.Es: # If the state has never been seen in the tree
            outcome = self.game.check_winner(board) 
            if outcome == 2:
                self.Es[s] = 1e-4
            else:
                self.Es[s] = outcome

        if self.Es[s] != 0:
            return -self.Es[s]


        # The game has not ended (not a leaf node)
        if s not in self.Ps: # No inference has been made on that node. We need to get the priors
            # leaf node
            self.Ps[s], v = self.net(torch.tensor(board).float().to(self.args['device']))
            self.Ps[s], v = self.Ps[s].detach().squeeze().cpu().numpy(), v.detach().squeeze().cpu().numpy()
            valids = np.array([1 if i in self.game.allowed_actions(board) else 0 for i in range(7) ])
            self.Ps[s] = self.Ps[s] * valids      # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                print("All valid moves were masked.")
                # We choose uniformely a move among the valid moves.
                self.Ps[s] += valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v
 
        # Expend with UCT : otherwise, 

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(7):
            if valids[a]:
                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + self.args['exploration_constant']*self.Ps[s][a]*np.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    u = self.args['exploration_constant']*self.Ps[s][a]*np.sqrt(self.Ns[s] + epsilon)

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.step(a, board, 1)
        next_s = self.game.get_canonical_form(next_s, next_player)

        v = self.search(next_s)

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return -v



