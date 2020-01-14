# Encode board for NN input

import numpy as np

def encode_board(env):
    board_state = env.current_state
    encoded = np.zeros([6,7,3]).astype(int)
    encoder_dict = {"0":0, "X":1}
    for row in range(6):
        for col in range(7):
            if board_state[row,col] != " ":
                encoded[row, col, encoder_dict[board_state[row,col]]] = 1
    if env.get_player_turn() == 1:
        encoded[:,:,2] = 1 # player to move
    return encoded

def decode_board(encoded):
    decoded = np.zeros([6,7]).astype(str)
    decoded[decoded == "0.0"] = " "
    decoder_dict = {0:"0", 1:"X"}
    for row in range(6):
        for col in range(7):
            for k in range(2):
                if encoded[row, col, k] == 1:
                    decoded[row, col] = decoder_dict[k]
    env = Connect4()
    env.current_state = decoded
    env.turn = (encoded[:,:,1] + encoded[:,:,0]).sum()
    return env
