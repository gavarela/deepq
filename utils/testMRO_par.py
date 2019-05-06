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

import numpy as np, random, json, os
from collections import deque

from WillyNet import WillyNet

from testSRO import QPlayer as QP_Base, play_hgame, play_cgame
from testMRO import Game, print_board


## QPlayer (one more method)
## ~~~~~~~

class QPlayer(QP_Base):
    
    def append_memory(self, mem):
        
        self.memory += mem
        
        while len(self.memory) > self.max_mem_len:
            self.memory.popleft()



## Play computer game (for parallel)
## ~~~~~~~~~~~~~~~~~~

def player_game(game, player, epsilon):
    
    t, w, _ = play_cgame(game, player, epsilon)
    return player.memory, t, w


## Do RL
## ~~~~~

if __name__ == "__main__":
    
    import time, matplotlib.pyplot as plt
    from pathos.multiprocessing import ProcessPool
    
    print('\033[33m')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('\n     Test - Simple RemainOne     \n')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('\033[0m')
    
    ## Params
    ## ~~~~~~

    SHAPE = [16, 
             400, 400, 
             len(Game._moves)]

    DISC_RATE = 0.95
    MAX_MEM_LEN = 1000

    BATCH_SIZE = 250
    L_RATE = 0.02
    REG_RATE = 0.0001
    MOM_RATE = 0.8
    
    N_PLAYERS = 6

    NUM_EPOCHS = 2000
    EPSILON = lambda i: 1.3 - i/NUM_EPOCHS

    VERBOSE = 25
    DET_VERBOSE = 50
    RAND_DETS = False
    
    SAVE_DIR = 'testMRO_saves/parallel1'
    
    
    # Play game - human
    # ~~~~~~~~~~~~~~~~~
    
    game = Game()
    
    if input('Enter anything to play the game: ') != '':
        
        print('\nWelcome to the game. Each turn, press 0, 1, 2, 3 to move to the right, left, up or down, respectively. Objective is to reach the food(x) with your player (A), without falling off the map or into a pit (O).')
        
        while True:
            
            play_hgame(game)
                
            if game.won:
                print('\nCongrats! You won :)')
            else:
                print('\nOh no! You lost :(')
            
            if input('Enter anything to play again: ') == '':
                break
    
    
    # Reinforcement learning
    # ~~~~~~~~~~~~~~~~~~~~~~
    
    print('\033[32m\n')
    print('Training via RL')
    print('~~~~~~~~~~~~~~~')
    print('\033[0m')
    
    print('Parameters:',
          '\n - discount rate       =', DISC_RATE,
          '\n - max memory len      =', MAX_MEM_LEN, '\n',
          '\n - batch size          =', BATCH_SIZE,
          '\n - learning rate       =', L_RATE,
          '\n - regularisation rate =', REG_RATE,
          '\n - momentum rate       =', MOM_RATE, '\n',
          '\n - epochs              =', NUM_EPOCHS, 
          '\n - epsilon(i)          = 1.3 - i/epochs', '\n')
    
    # Games and players
    master = QPlayer(MAX_MEM_LEN, DISC_RATE, SHAPE)
    players = []
    games = []
    for i in range(N_PLAYERS):
        players.append(QPlayer(MAX_MEM_LEN, DISC_RATE, SHAPE))
        games.append(Game())
    
    turnlist = []
    winlist = []
    
    det_turnlist = []
    det_winlist = []
    
    last_hist = ''
    
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        
        # Reset players
        for i in range(len(players)):
            players[i].memory = deque()
            players[i].network = master.network.copy()
        
        # Play games
        pool = ProcessPool(N_PLAYERS)
        output = pool.map(lambda g, p: player_game(g, p, EPSILON(epoch)),
                          games, players)
        
        turns = []
        wins = []
        for mem, turn, won in output:
            master.append_memory(mem)
            turns.append(turn)
            wins.append(won)
        
        turnlist += turns
        winlist += wins
        
        if (epoch+1) % VERBOSE == 0:
            print('\nEpoch %i: played %i games w/ mean turns %0.1f and %i win(s).' %(epoch+1, N_PLAYERS, sum(turns)/N_PLAYERS, sum(wins)))
        
        # Train
        master.train(len(master.memory), L_RATE, REG_RATE, MOM_RATE)
        master.memory = deque()
        
        # If right epoch, play deterministic game
        if (epoch+1) % DET_VERBOSE == 0:
            
            turns, won, this_hist = play_cgame(game, master, -1, RAND_DETS)
            
            print('\n  Played a deterministic game %i minutes into training.\n  Lasted %i turns and %s' \
                  %((time.time() - start_time) / 60,
                    turns,
                    'won!' if won else 'lost...')
                 )
            
            print('    Hist:', this_hist)
            if this_hist == last_hist:
                print('           PLAYED THE SAME GAME!!!')
            last_hist = this_hist
            
            det_turnlist.append(11 - turns)
            det_winlist.append(won)
    
    print('\nDone running RL! Saving...')
    
    # All done, save network and turn list and stuff
    master.network.save(SAVE_DIR + '/network.json')
    
    data = {'turnlist': turnlist,
            'winlist': winlist,
            'det_turnlist': det_turnlist,
            'det_winlist': det_winlist,
            'params': {'DISC_RATE': DISC_RATE,
                       'MAX_MEM_LEN': MAX_MEM_LEN,
                       'BATCH_SIZE': BATCH_SIZE,
                       'L_RATE': L_RATE,
                       'REG_RATE': REG_RATE,
                       'MOM_RATE': MOM_RATE,
                       'N_PLAYERS': N_PLAYERS,
                       'NUM_EPOCHS': NUM_EPOCHS,
                       'DET_VERBOSE': DET_VERBOSE}
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
    ax.set_ylabel('Turns Played')
    ax.set_title('RL Progress')
    
    text = 'learn: %0.4f \n' \
           'reg  : %0.4f \n' \
           'mom  : %0.2f \n' \
           'batch: %i \n' \
            % (L_RATE, REG_RATE, MOM_RATE, BATCH_SIZE)

    ax.text(0.98, 0.98, text,
            transform = ax.transAxes, fontsize = 9,
            verticalalignment = 'top',
            horizontalalignment = 'right')
    
    plt.show(block = False)
    plt.savefig(SAVE_DIR + '/progress.pdf')
    input('PE.')
    plt.close()
            
        
        
        
    
    
    
    
    
    
    
    
    
    
        
    
    

        
    