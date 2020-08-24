import numpy as np
import torch

class Node:
    def __init__(self, state, turn):
        self.id = str(state)
        self.state = state
        self.turn = turn
        self.edges = []
        
    def is_leaf(self):
        return len(self.edges) == 0
    
class Edge:
    def __init__(self, parent, child, prior, action):
        self.id = (parent.id, str(action))
        self.parent = parent
        self.child = child
        self.turn = parent.turn
        self.action = action
        
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = prior


class MCTS:
    def __init__(self, root, turn, game, net, args):
        self.root = Node(root, turn)
        self.game = game
        self.net = net
        self.args = args
        self.tree = {}
        self.tree[str(root)] = self.root
        
    
    def select_leaf(self):
        branch = []
        current_node = self.root
        
        while current_node.is_leaf() == False:
            best_qu = -np.inf
            best_edge = None

            Nb = 0
            for edge in current_node.edges:
                Nb += edge.N
            
            if Nb == 0:
                Nb += 1
            
            # Adding dirichlet noise to the root node for exploration
            if current_node == self.root:
                nu = np.random.dirichlet([self.args['alpha']] * len(current_node.edges))
                epsilon = self.args['epsilon']
                
            else:
                epsilon = 0
                nu = [0] * len(current_node.edges)

            # select the next node
            for i, edge in enumerate(current_node.edges):
                u = self.args['cpuct'] * ((1-epsilon) * edge.P + epsilon * nu[i]) * np.sqrt(Nb) / (1 + edge.N)
                q = edge.Q
            
                if q + u > best_qu:
                    best_qu = q + u
                    best_edge = edge
#                print(i,"q" , q, "u", u, "Nb", Nb)
                
            branch.append(best_edge)
            current_node = best_edge.child
        
        winner = self.game.check_winner(current_node.state)
        
        return current_node, winner, branch
    
    def expand_evaluate(self, leaf, winner):
        if winner == 1:
            return 1.
        elif winner == -1:
            return -1.
        elif winner == 2:
            return 1e-4
        else:
            probs, value = self.net(torch.tensor(leaf.state).float().to(self.args['device']))
            probs, value = probs.detach().squeeze().cpu().numpy(), value.detach().squeeze().cpu().numpy()
            valid_actions = np.array([1 if i in self.game.allowed_actions(leaf.state) else 0 for i in range(7)])
            # Masking invalid moves and normalize
            probs = probs * valid_actions
            probs = probs / np.sum(probs)
            
            # Expand the tree according to all possible actions
            for i, a in enumerate(self.game.allowed_actions(leaf.state)):
                next_state, next_turn = self.game.step(a, leaf.state, leaf.turn)
                if str(next_state) not in self.tree.keys():
                    node = Node(next_state, next_turn)
                    self.tree[str(next_state)] = node
                else:
                    node = self.tree[str(next_state)]
                
                new_edge = Edge(leaf, node, probs[i], a)
                leaf.edges.append(new_edge)
        return value

    def backup(self, leaf, value, branch):
        leaf_turn = leaf.turn
#        print("leaf turn", leaf_turn)
        for edge in branch:
            turn = edge.turn
            if turn == leaf_turn:
                sgn = 1
            else:
                sgn = -1
                
            edge.N += 1
            edge.W += sgn * value
            edge.Q = edge.W / edge.N
    
    def search(self):
        # Select the branch
        leaf, winner, branch = self.select_leaf()
        # Evaluate and expand
        value = self.expand_evaluate(leaf, winner)
        # Back up the value in the branch
        self.backup(leaf, value, branch)
    
    def get_action_probs(self, tau):
        for _ in range(self.args['n_sim']):
            self.search()
        edges = self.root.edges
        pi = [0.] * 7
        values = [0.] * 7
        for edge in edges:
            if tau != 0.:
                pi[edge.action] = edge.N ** (1/tau)
            else:
                pi[edge.action] = np.float(edge.N)
            
            values[edge.action] = edge.Q
        pi = np.array(pi)

        if np.sum(pi) != 0:
            pi /= np.sum(pi)
        else:
            a = np.random.choice(self.game.get_allowed_actions(self.root.state))
            pi[a] = 1
        return pi, values
    
    def set_root(self, node):
        if str(node) in self.tree.keys():
            self.root = self.tree[str(node)]
        else:
            if node.sum() == 0:
                turn = 1
            else:
                turn = -1
            self.root = Node(node, turn)
            self.tree[str(self.root)] = self.root

