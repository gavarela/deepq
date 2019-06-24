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

from testSRO import QPlayer
from testMRO import Game, print_board, play_hgame


## Policy Gradient Ascent Player
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def one_hot(A, n_classes):
    return np.eye(n_classes)[A]

class PolPlayer(QPlayer):
    
    def __init__(self, max_mem_len, disc_rate,
                 state_size, hidden_shape, n_actions):
        
        self.max_mem_len = max_mem_len
        self.disc_rate = disc_rate
        
        self.memory = deque()
        
        self.trains = 0
        
        # Network
        self.state_size = state_size
        self.n_actions = n_actions
        
        shape = [state_size] + hidden_shape + [n_actions]
        self.network = WillyNet(shape = shape,
                                problem = 'softmax')
    
    def get_action(self, state, at_random, legal_moves):
        
        lm_inds = np.where(legal_moves)[0]
        
        a_probs = self.network.predict(np.array([state]))
        
        legal_probs = a_probs[0, lm_inds]
        
        if at_random:
            return random.choice(lm_inds)
            #entropy = -sum(a_probs * np.log(a_probs))
            legal_probs /= sum(legal_probs)
            return np.random.choice(lm_inds, p = legal_probs)
        
        else:
            return lm_inds[legal_probs.argmax()]
    
    def train(self, mem_batch, batch_prop, 
              l_rate, reg_rate, mom_rate):
        
        self.trains += 1
        
        # Get batch
        if mem_batch < len(self.memory):
            batch = np.array(random.sample(self.memory, mem_batch))
        else:
            batch = np.array(self.memory)
        
        # Build examples and targets
        state, action, Reward = \
            [np.array(list(batch[:, i])).reshape((len(batch), -1)) for i in range(len(batch[0]))]
        
        a_probs = self.network.predict(state)
        a_taken = one_hot(action.reshape(-1), self.n_actions)
        
        X = state
        y = a_taken
        W = -np.tile(Reward, self.n_actions)
            # minus for gradient ascent on policy^
        
        # Train
        self.network.train(X, y,
                           weights = W,
                           num_iterations = int(1/batch_prop),
                           batch_size = int(batch_prop*mem_batch),
                           learn_rate = l_rate,
                           reg_rate = reg_rate,
                           mom_rate = mom_rate)
    
    def play_game(self, game, epsilon, rand_start = False):
        
        game.restart(rand_start)
        
        hist = ''
        turns = 0
        temp_mem = []
        new_state = game.get_state()
        while not (game.crashed or game.won):
            
            # Play turn
            state = new_state
            lms = game.lms
            
            action = self.get_action(state,
                                     random.random() < epsilon,
                                     lms)
            
            reward = game.turn(action)
            new_state = game.get_state()
            
            turns += 1
            
            # Store in memory
            temp_mem.append((state, action, reward))
            
            # History
            hist += str(action) + ' '
        
        R = 0
        for s, a, r in reversed(temp_mem):
            R = r + self.disc_rate*R
            self.store(s, a, R)

        return turns, game.won, hist


## Do RL
## ~~~~~

if __name__ == "__main__":
    
    import time, matplotlib.pyplot as plt
    
    print('\033[33m')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('\n     Test - Medium RemainOne     \n')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('\033[0m')
    
    ## Params
    ## ~~~~~~

    STATE_SIZE = 16
    HIDDEN_SHAPE = [400, 600]
    N_ACTIONS = len(Game._moves)

    DISC_RATE = 0.95
    MAX_MEM_LEN = 1000

    MEM_BATCH = 150
    TRAIN_BATCH = 1/5
    
    L_RATE = 0.00001 # * MEM_BATCH * TRAIN_BATCH
    REG_RATE = 0.0005
    MOM_RATE = 0.8

    NUM_EPOCHS = 10000
    EPSILON = lambda i: 1.3 - i/NUM_EPOCHS

    VERBOSE = 25
    DET_VERBOSE = 50
    RAND_DETS = False
    
    SAVE_DIR = 'testMRO_saves/polgrad/e%i_m%i_t%i_l%s_r%s' %(NUM_EPOCHS, MEM_BATCH, TRAIN_BATCH * MEM_BATCH, str(L_RATE).replace('.', '-'), str(REG_RATE).replace('.', '-'))
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
    
    player = PolPlayer(MAX_MEM_LEN, DISC_RATE,
                       STATE_SIZE, HIDDEN_SHAPE, N_ACTIONS)
    
    turnlist = []
    winlist = []
    
    det_turnlist = []
    det_winlist = []
    
    last_hist = ''
    
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        
        # Play a game
        turns, won, _ = player.play_game(game, EPSILON(epoch))
        
        turnlist.append(turns)
        winlist.append(game.won)
        
        if (epoch+1) % VERBOSE == 0:
            print('\nEpoch %i: played %i turn(s) and %s' %(epoch+1, turns, 'won!' if won else 'lost...'))
        
        # Train
        player.train(MEM_BATCH, TRAIN_BATCH, 
                     L_RATE, REG_RATE, MOM_RATE)
        
        # If right epoch, play deterministic game
        if (epoch+1) % DET_VERBOSE == 0:
            
            turns, won, this_hist = player.play_game(game, -1, RAND_DETS)
            
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
            'time': end_time - start_time}
    
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
               MEM_BATCH, TRAIN_BATCH * MEM_BATCH,
               (end_time - start_time)/60)

    ax.text(0.98, 0.98, text,
            transform = ax.transAxes, fontsize = 9,
            verticalalignment = 'top',
            horizontalalignment = 'right')
    
    plt.show(block = False)
    plt.savefig(SAVE_DIR + '/progress.pdf')
    input('PE.')
    plt.close()
        
    
    
    
    
    
    
    
    
    
        
    
    

        
    