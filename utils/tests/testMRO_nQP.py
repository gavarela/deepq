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


## n-step QPlayer
## ~~~~~~~~~~~~~~

class nQPlayer(QPlayer):
    
    def train(self, mem_batch, batch_prop, 
              l_rate, reg_rate, mom_rate):
        
        self.trains += 1
        
        # Get batch
        if mem_batch < len(self.memory):
            batch = np.array(random.sample(self.memory, mem_batch))
        else:
            batch = np.array(self.memory)
        
        # Build examples and targets
        state, action, Reward, legal_moves = \
            [np.array(list(batch[:, i])).reshape((len(batch), -1)) for i in range(len(batch[0]))]
        
        examples = state
        
        targets = self.network.predict(state)
        targets *= legal_moves
        
        action_inds = np.tile(range(targets.shape[1]), (targets.shape[0], 1)) == action
        
        targets += action_inds * (-targets + Reward)
        
        # Train
        self.network.train(np.array(examples),
                           np.array(targets),
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
            temp_mem.append((state, action, reward, lms))

            # History
            hist += str(action) + ' '
        
        R = 0
        for s, a, r, lms in reversed(temp_mem):
            R = r + self.disc_rate*R
            self.store(s, a, R, lms)

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

    SHAPE = [16, 
             400, 400, 
             len(Game._moves)]

    DISC_RATE = 0.95
    MAX_MEM_LEN = 1000

    MEM_BATCH = 150
    TRAIN_BATCH = 1/5
    
    L_RATE = 0.5 # 0.02 * MEM_BATCH * TRAIN_BATCH
    REG_RATE = 0.0001
    MOM_RATE = 0.8

    NUM_EPOCHS = 3000
    EPSILON = lambda i: 1.3 - i/NUM_EPOCHS

    VERBOSE = 25
    DET_VERBOSE = 50
    RAND_DETS = False
    
    SAVE_DIR = 'testMRO_saves/nstep/ e%i_m%i_t%i_l%s' %(NUM_EPOCHS, MEM_BATCH, TRAIN_BATCH * MEM_BATCH, str(L_RATE).replace('.', '-'))
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
    
    player = nQPlayer(MAX_MEM_LEN, DISC_RATE, SHAPE)
    
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
        
    
    
    
    
    
    
    
    
    
        
    
    

        
    