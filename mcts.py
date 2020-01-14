import numpy as np
from encode import encode_board
from copy import deepcopy
import torch

epsilon=1e-7

class MCTS:
    """
    Monte Carlo Tree search algorithm
    """

    def __init__(self, net, args, device="cpu"):
        self.net = net
        self.args = args
        self.device = device
        self.Qsa = {}       # Q values
        self.Nsa = {}       # number of times edge s,a was visited
        self.Ns = {}        # # number of times board s was visited
        self.Ps = {}        # initial policy

        self.Es = {}        # game status for board s
        self.Vs = {}        # valid actions for board s

    def run_sims(self, root_board, scale=1.):
        for i in range(self.args.n_sim):
            self.Vs[str(root_board.current_state)] = self.search(root_board)

        s = str(root_board.current_state)
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(7)]

        if scale == 0.:
            best_action = np.argmax(counts)
            policy = [0]*len(counts)
            policy[best_action] = 1
            return policy

        counts = [x**(1./scale) for x in counts]
        counts_sum = np.sum(counts)
        policy = [x/counts_sum for x in counts]
        return policy

    def search(self, root_board):
        rb = deepcopy(root_board)
        s = str(rb.current_state)
        if s not in self.Es:
            status = 0
            if rb.winner == "0":
                self.Es[s] = 1
            elif rb.winner == "X":
                self.Es[s] = -1
            elif rb.winner == 'draw':
                self.Es[s] = 1e-3
            else:
                self.Es[s] = 0

        if self.Es[s]!= 0:
            # The node is terminal
            return -self.Es[s]


        # The game has not ended (not a leaf node)
        if s not in self.Ps: # No inference has been made on that node. We need to get the priors
            # leaf node
            print(encode_board(rb).shape)
            print(rb.current_state)
            self.Ps[s], v = self.net(torch.tensor(encode_board(rb)).float().unsqueeze(0))
            self.Ps[s], v = self.Ps[s].detach().numpy(), v.detach().numpy()
            valids = np.array([1 if i in rb.allowed_actions() else 0 for i in range(7) ])
            self.Ps[s] = self.Ps[s] * valids      # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                print("All valid moves were masked.")
                self.Ps[s] = self.Ps[s] + valids
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
                    u = self.Qsa[(s,a)] + self.args.exploration_constant*self.Ps[s][a]*np.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    u = self.args.exploration_constant*self.Ps[s][a]*np.sqrt(self.Ns[s] + epsilon)     # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        rb.step(a)
        next_s = rb.to_canonical()

        v = self.search(next_s)

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return -v



