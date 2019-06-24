## ~~~~~~~~~~~~~~~~~~
#  Testing Q-Learning
#  Small Remain One
## ~~~~~~~~~~~~~~~~~~


'''

IF WANT TO CHANGE BOARD SIZE, FOLLOWING CHANGES NEED TO BE MADE:
 - BOARD, ROW_RANGE and COL_RANGE
 - In Game __init__: self.board, self.n_pieces and self.do_first_move stuff
 - In print_board: Indexing (abc and 123)
 - In det game bit: Score appended to det_turnlist

'''

import sys
if "../" not in sys.path:
    sys.path.append("../")

import numpy as np, random, json, os
from collections import deque

from WillyNet import WillyNet

import keras
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, Flatten, Dense, Dropout

from testSRO import QPlayer as QP_Base, play_cgame
from testMRO import Game as Game_Base, print_board, play_hgame


## Game (2d state)
## ~~~~

class Game(Game_Base):
    
    def get_state(self):
        
        state = np.copy(self.board)[2:-2, 2:-2]
        #state = np.where(state == 2, 0, state)
        
        return state.reshape(list(state.shape) + [1])


## Conv Net
## ~~~~~~~~

class NiceKeras(object):
    
    def __init__(self, in_shape, n_conv, n_dense, n_out,
                 l_rate, momentum):
        
        print(in_shape, n_conv, n_dense, n_out)
        
        self.network = Sequential()
        self.network.add(Conv2D(n_conv, kernel_size = (3, 3), input_shape = in_shape, activation = 'relu'))
        self.network.add(Flatten())
        #self.network.add(Dropout(0.1))
        self.network.add(Dense(n_dense, activation = 'relu'))
        self.network.add(Dropout(0.2))
        self.network.add(Dense(n_out, activation = 'linear'))

        self.network.compile(
            loss = keras.losses.mean_squared_error, 
            optimizer = keras.optimizers.SGD(lr = l_rate, momentum = momentum),
            metrics = [keras.losses.mean_squared_error])
    
    def train(self, X, y, num_iterations, batch_size,
              learn_rate, reg_rate, mom_rate):
        
        self.network.fit(X, y,
                         batch_size = batch_size,
                         epochs = num_iterations,
                         verbose = 0)
    
    def predict(self, X):
        
        return self.network.predict(X)
    
    def save(self, filename):
        
        with open(filename, 'w') as file:
            file.write(self.network.to_json())
        self.network.save_weights(filename.replace('json', 'h5'))
    
    @classmethod
    def load(cls, filename):
        
        # Null class object
        class Null(object):
            pass
        
        instance = Null()
        
        # Create network
        with open(filename, 'r') as file:
            instance.network = model_from_json(file.read())
        
        # Get weights
        self.network.load_weights(filename.replace('json', 'h5'))
        
        # Set class and return
        instance.__class__ = cls
        return instance
        

## QPlayer (w/ conv net)
## ~~~~~~~

class QPlayer(QP_Base):
    
    def __init__(self, max_mem_len, disc_rate, shape,
                 l_rate, mom_rate):
        
        self.max_mem_len = max_mem_len
        self.disc_rate = disc_rate

        self.memory = deque()

        self.network = NiceKeras(*shape, l_rate, mom_rate)
        self.trains = 0
        
        




## Do RL
## ~~~~~

if __name__ == "__main__":
    
    import time, matplotlib.pyplot as plt
    
    print('\033[33m')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('\n     Test - Simple RemainOne     \n')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('\033[0m')
    
    ## Params
    ## ~~~~~~

    SHAPE = [(4, 5, 1), 
             20, 200, 
             len(Game._moves)]

    DISC_RATE = 0.95
    MAX_MEM_LEN = 1000

    MEM_BATCH = 150
    TRAIN_BATCH = 1/75
    
    L_RATE = 0.02
    REG_RATE = 0.0001
    MOM_RATE = 0.8

    NUM_EPOCHS = 3000
    EPSILON = lambda i: 1.3 - i/NUM_EPOCHS

    VERBOSE = 25
    DET_VERBOSE = 50
    RAND_DETS = False
    
    SAVE_DIR = 'testMRO_saves/conv_batches/e%i_m%i_t%i' %(NUM_EPOCHS, MEM_BATCH, TRAIN_BATCH*MEM_BATCH)
    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    
    
    game = Game()
    
    
    # Play game - human
    # ~~~~~~~~~~~~~~~~~
    
    play = input('Enter anything to play the game: ')
    while play != '':
        play_hgame(game)
        play = input('Enter anything to play again: ')
    
    
    # Reinforcement learning
    # ~~~~~~~~~~~~~~~~~~~~~~
    
    print('\033[32m\n')
    print('Training via RL')
    print('~~~~~~~~~~~~~~~')
    print('\033[0m')
    
    print('Parameters:',
          '\n - discount rate       =', DISC_RATE,
          '\n - max memory len      =', MAX_MEM_LEN, '\n',
          '\n - memory batch size   =', MEM_BATCH,
          '\n - training batch size =', TRAIN_BATCH, '\n',
          '\n - learning rate       =', L_RATE,
          '\n - regularisation rate =', REG_RATE,
          '\n - momentum rate       =', MOM_RATE, '\n',
          '\n - epochs              =', NUM_EPOCHS, 
          '\n - epsilon(i)          = 1.3 - i/epochs', '\n')
    
    player = QPlayer(MAX_MEM_LEN, DISC_RATE, SHAPE, L_RATE, MOM_RATE)
    
    turnlist = []
    winlist = []
    
    det_turnlist = []
    det_winlist = []
    
    last_hist = ''
    
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        
        # Play a game
        turns, won, _ = play_cgame(game, player, EPSILON(epoch))
        
        turnlist.append(turns)
        winlist.append(game.won)
        
        if (epoch+1) % VERBOSE == 0:
            print('\nEpoch %i: played %i turn(s) and %s' %(epoch+1, turns, 'won!' if won else 'lost...'))
        
        # Train
        player.train(MEM_BATCH, TRAIN_BATCH, 
                     L_RATE, REG_RATE, MOM_RATE)
        
        # If right epoch, play deterministic game
        if (epoch+1) % DET_VERBOSE == 0:
            
            turns, won, this_hist = play_cgame(game, player, -1, RAND_DETS)
            
            print('\n  Played a deterministic game %i minutes into training.\n  Lasted %i turns and %s' \
                  %((time.time() - start_time) / 60,
                    turns,
                    'won!' if won else 'lost...')
                 )
            
            print('    Hist:', this_hist)
            if this_hist == last_hist:
                print('           PLAYED THE SAME GAME!!!')
            last_hist = this_hist
            
            det_turnlist.append(15 - turns)
            det_winlist.append(won)
    
    end_time = time.time()
    print('\nDone running RL! Saving...')
    
    # All done, save network and turn list and stuff
    player.network.save(SAVE_DIR + '/network.json')
    
    data = {'turnlist': turnlist,
            'winlist': winlist,
            'det_turnlist': det_turnlist,
            'det_winlist': det_winlist,
            'params': {'DISC_RATE': DISC_RATE,
                       'MAX_MEM_LEN': MAX_MEM_LEN,
                       'MEM_BATCH': MEM_BATCH,
                       'TRAIN_BATCH': TRAIN_BATCH,
                       'L_RATE': L_RATE,
                       'REG_RATE': REG_RATE,
                       'MOM_RATE': MOM_RATE,
                       'NUM_EPOCHS': NUM_EPOCHS,
                       'DET_VERBOSE': DET_VERBOSE},
            'time': end_time - start_time,
            }
    
    file = open(SAVE_DIR + '/results.json', 'w')
    json.dump(data, file)
    file.close()
    
    # Plot
    print('Saved. Plotting...')
    fig, ax = plt.subplots()
    
    for i in range(len(det_winlist)):
        ax.plot(i * DET_VERBOSE, det_turnlist[i], 
                'g.' if det_winlist[i] else 'r.',
                markersize = 1)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pieces Left')
    ax.set_title('RL Progress')
    
    text = 'learn: %0.4f \n' \
           'reg: %0.4f \n' \
           'mom: %0.2f \n\n' \
           'm batch: %i \n' \
           't batch: %i \n\n' \
           'time: %0.1f mins' \
            % (L_RATE, REG_RATE, MOM_RATE, 
               MEM_BATCH, TRAIN_BATCH*MEM_BATCH,
               (end_time - start_time)/60)

    ax.text(0.98, 0.98, text,
            transform = ax.transAxes, fontsize = 9,
            verticalalignment = 'top',
            horizontalalignment = 'right')
    
    plt.show(block = False)
    plt.savefig(SAVE_DIR + '/progress.pdf')
    input('PE.')
    plt.close()
        
    
    
    
    
    
    
    
    
    
        
    
    

        
    