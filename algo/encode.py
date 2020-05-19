# Encode board for NN input

import numpy as np

def encode_board(board_state, player_turn=1):
    encoded = np.zeros([6,7,3]).astype(int)
    encoder_dict = {-1:0, 1:1}
    for row in range(6):
        for col in range(7):
            if board_state[row,col] != 0:
                encoded[row, col, encoder_dict[board_state[row,col]]] = 1
    if player_turn == 1:
        encoded[:,:,2] = 1 # player to move
    return encoded

def decode_board(encoded):
    decoded = np.zeros([6,7])
    decoder_dict = {0:-1, 1:1}
    for row in range(6):
        for col in range(7):
            for k in range(2):
                if encoded[row, col, k] == 1:
                    decoded[row, col] = decoder_dict[k]
    turn = (encoded[:,:,1] + encoded[:,:,0]).sum()
    return decoded, -1**turn

def encode_batch(examples):
    """ Examples take form : [(s, p, v)] """
    