import numpy as np
from tqdm import tqdm

class Arena:

    def __init__(self, mcts1, mcts2, game):
        """ Player 1 begins """
        self.player1 = player1
        self.player2 = player2
        self.game = game

    def play_game(self, verbose=False, start=1, store_game=False): # start = {-1, 1}
        self.players = {int(start):self.player1, int(-1*start):self.player2}
        curPlayer = 1
        board = self.game.init_state
        it = 0
        action = None
        while self.game.check_winner(board)==0:
            it+=1
            if verbose:
                print("Turn ", str(it), "Previous action ", str(action),  "Player ", str(curPlayer))
                print(self.game.render(board))
            action = self.players[curPlayer](self.game.get_canonical_form(board, curPlayer))

            valids = self.game.allowed_actions(board)

            if action in valids:
                board, curPlayer = self.game.step(action, board, curPlayer)
        if verbose:
            print("Last Turn ", str(it), "Result ", str(self.game.check_winner(board)))
            self.game.render(board)

        return start * self.game.check_winner(board)

    def evaluate(self, num):
        stats = {}
        
        stats['one_won'] = 0
        stats['two_won'] = 0
        stats['draws'] = 0

        for _ in tqdm(range(num//2)):
            result = self.play_game(verbose=False,start=1)
            if result==1:
                stats['one_won']+=1
            elif result==-1:
                stats['two_won']+=1
            else:
                stats['draws']+=1

            result = self.play_game(verbose=False,start=-1)
            if result==1:
                stats['one_won']+=1
            elif result==-1:
                stats['two_won']+=1
            else:
                stats['draws']+=1

        return stats
