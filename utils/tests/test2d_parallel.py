## ~~~~~~~~~~~~~~~~~~
#  Testing Q-Learning
## ~~~~~~~~~~~~~~~~~~


'''

Simple game. Build grid and randomly place player, food and (if set to) pits. Want to get to food without falling in pits or off the grid. Easy to solve when small. Gets harder fast...

Params we can solve with:
B = 3, P = 0: using l_rate 0.02 and epochs 700
B = 3, P = 1: using l_rate 0.02 and epochs 1500
B = 3, P = 2: epochs 2500, maybe more
B = 4, P = 0:     (same as above ^)
B = 4, P = 1: epochs 4500 gets almost there
B = 5, P = 0: epochs 4500

'''

import numpy as np, random, json
from collections import deque

from WillyNet import WillyNet

import multiprocessing as mp
from pathos.multiprocessing import ProcessPool

from test2d import Game, QPlayer, play_hgame


## Copy method for QPlayer
## ~~~~~~~~~~~~~~~~~~~~~~~

def copy(self):
    
    copy = QPlayer(self.max_mem_len, self.disc_rate, self.network.shape)
    
    copy.trains = self.trains
    
    copy.memory = self.memory.copy()
    copy.network = self.network.copy()
    
    return copy

QPlayer.copy = copy


## Play game (computer)
## ~~~~~~~~~

def play_cgame(game, player, epsilon):
    
    #print('  In cgame, player has', player.trains, 'trains')
    
    game.restart()
    
    turns = 0
    memory = deque()
    
    new_state = game.get_state()
    while (not (game.crashed or game.won)) and (turns < MAX_TURNS):

        # Play turn
        state = new_state
        action = player.get_action(state, 
                                   random.random() < epsilon)
        reward = game.turn(action)
        new_state = game.get_state()

        turns += 1
        
        # Store in memory
        memory.append([state, action, reward, new_state, game.crashed])
    
    return turns, game.won, memory

def play_cgames(n_workers, games, player, epsilon):
    
    worker_pool = ProcessPool(n_workers)
    output = worker_pool.map(lambda g: play_cgame(g, player.copy(), epsilon), games)
    
    turns = []
    wins = []
    memory = []
    for _turns, _win, _memory in output:

        turnlist.append(_turns)
        winlist.append(_win)
        for item in _memory:
            memory.append(item)
    
    return turns, wins, memory


def pcg(game, player, epsilon):
    
    game.restart()
    
    turns = 0
    new_state = game.get_state()
    while (not (game.crashed or game.won)) and (turns < MAX_TURNS):

        # Play turn
        state = new_state
        action = player.get_action(state, 
                                   random.random() < epsilon)
        reward = game.turn(action)
        new_state = game.get_state()

        turns += 1
        
        # Store in memory
        player.store(state, action, reward, new_state, game.crashed)
    
    return turns, game.won

## Do RL
## ~~~~~

if __name__ == "__main__":
    
    import time, matplotlib.pyplot as plt
    
    print('\033[33m')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('\n       Test - Simple Game       \n')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('\033[0m')
    
    ## Params
    ## ~~~~~~

    N_PITS = 0
    BOARD_SIZE = 3

    SHAPE = [(2 + (N_PITS>0))*BOARD_SIZE**2, 
             200, 200, 
             4]

    DISC_RATE = 0.95
    MAX_MEM_LEN = 1000

    BATCH_SIZE = 70
    L_RATE = 0.008
    REG_RATE = 0.0001
    MOM_RATE = 0.8

    N_WORKERS = 6
    NUM_EPOCHS = 100
    EPSILON = lambda i: 1 - i/NUM_EPOCHS

    VERBOSE = 10
    DET_VERBOSE = 1
    NUM_DET_GAMES = 15
    MAX_DET_TRIES = 50

    MAX_TURNS = 100
    
    # Play game - human
    # ~~~~~~~~~~~~~~~~~
    
    if input('Enter anything to play the game: ') != '':
        
        game = Game(N_PITS, BOARD_SIZE)
        
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
    
    player = QPlayer(MAX_MEM_LEN, DISC_RATE, SHAPE)
    
    game_pool = []
    for g in range(N_WORKERS):
        game_pool.append(Game(N_PITS, BOARD_SIZE))
    
    print('\033[32m\n')
    print('Training via RL')
    print('~~~~~~~~~~~~~~~')
    print('\033[0m')
    
    print('Parameters:',
          '\n - Board size : %i x %i' %(BOARD_SIZE, BOARD_SIZE),
          '\n - # pits     : %i' % N_PITS, '\n',
          '\n - discount rate       =', DISC_RATE,
          '\n - max memory len      =', MAX_MEM_LEN, '\n',
          '\n - batch size          =', BATCH_SIZE,
          '\n - learning rate       =', L_RATE,
          '\n - regularisation rate =', REG_RATE,
          '\n - momentum rate       =', MOM_RATE, '\n',
          '\n - epochs              =', NUM_EPOCHS, 
          '\n - epsilon(i)          = 1 - i/epochs', '\n')
    
    turnlist = []
    winlist = []
    
    det_winlist = []
    det_trylist = []
    
    game = Game(N_PITS, BOARD_SIZE)
    
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        
        # Have workers play games
        turns, wins, memory = play_cgames(N_WORKERS, game_pool, player, EPSILON(epoch))
        
        turnlist += turns
        winlist += wins
        for item in memory:
            player.store(*item)
        
        # Report
        if (epoch+1) % VERBOSE == 0:
            print('\nEpoch %i: played %i games and won %i.' %(epoch+1, N_WORKERS, sum(wins)))
        
        # Train
        player.train(BATCH_SIZE, L_RATE, REG_RATE, MOM_RATE)
        #print('\nEpoch', epoch+1, '\nPlayer has', player.trains, 'trains.')
        #print('Copy has', player.copy().trains, 'trains.\n')
        
        # If right epoch, play deterministic games
        if (epoch+1) % DET_VERBOSE == 0:
            
            print('\n  Det games:')
            
            # Play three that don't get stuck
            n_wins = 0
            non_stuck = 0
            tries = 0
            while non_stuck < NUM_DET_GAMES and tries < MAX_DET_TRIES:
                
                turns, won = pcg(game, player, -1)
                
                if turns < MAX_TURNS:
                    non_stuck += 1
                    n_wins += won
                tries += 1
            
            print('  Played %i deterministic games %i minutes into training.\n    %i got stuck. Of the %i that did not, won %i of them.' \
                  %(tries, 
                    (time.time() - start_time) / 60,
                    tries - non_stuck,
                    NUM_DET_GAMES,
                    n_wins)
                 )
            
            det_winlist.append(n_wins)
            det_trylist.append(15)
            
#            # Play three that don't get stuck
#            n_wins = 0
#            non_stuck = 0
#            tries = 0
#            while non_stuck < NUM_DET_GAMES and tries < MAX_DET_TRIES:
#                
#                output = worker_pool.map(lambda g: play_cgame(g, player.copy(), -1), game_pool)
#                
#                tries += N_WORKERS
#                
#                for _, wins, _ in output:
#                    
#                    non_stuck += len(np.where(turns < MAX_TURNS)[0])
#                    n_wins += len(np.where(wins)[0])
#                    
#                    #if non_stuck >= NUM_DET_GAMES: break
#            
#            #n_wins = min(NUM_DET_GAMES, n_wins)
#            
#            print('  Played %i deterministic games %i minutes into training.\n    %i got stuck. Of the %i that did not, won %i of them (%0.1f).' \
#                  %(tries, 
#                    (time.time() - start_time) / 60,
#                    tries - non_stuck,
#                    non_stuck,
#                    n_wins,
#                    n_wins/non_stuck)
#                 )
#            
#            det_winlist.append(n_wins)
#            det_trylist.append(non_stuck)
            
    print('\nDone running RL! Saving...')
    
    # All done, save network and turn list and stuff
    player.network.save('test_saves/network_%i_%i_para.json' %(BOARD_SIZE, N_PITS))
    
    data = {'turnlist': turnlist,
            'winlist': winlist,
            'det_winlist': det_winlist,
            'det_trylist': det_trylist,
            'params': {'N_PITS': N_PITS,
                       'BOARD_SIZE': BOARD_SIZE,
                       'DISC_RATE': DISC_RATE,
                       'MAX_MEM_LEN': MAX_MEM_LEN,
                       'BATCH_SIZE': BATCH_SIZE,
                       'L_RATE': L_RATE,
                       'REG_RATE': REG_RATE,
                       'MOM_RATE': MOM_RATE,
                       'N_WORKERS': N_WORKERS,
                       'NUM_EPOCHS': NUM_EPOCHS,
                       'DET_VERBOSE': DET_VERBOSE,
                       'NUM_DET_GAMES': NUM_DET_GAMES,
                       'MAX_DET_TRIES': MAX_DET_TRIES,
                       'MAX_TURNS': MAX_TURNS}
            }
    
    file = open('test_saves/results_%i_%i_para.json' %(BOARD_SIZE, N_PITS), 'w')
    json.dump(data, file)
    file.close()
    
    # Plot
    print('Saved. Plotting...')
    fig, ax = plt.subplots()
    
    for i in range(len(det_winlist)):
        ax.plot(i * DET_VERBOSE, det_winlist[i]/det_trylist[i], 
                'g.' if det_winlist[i]/det_trylist[i] > 0.5 else 'r.',
                markersize = 1)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Games won (proportion)')
    ax.set_title('(%i, %i) w/ %i pit(s)' %(BOARD_SIZE, BOARD_SIZE,
                                           N_PITS))
    
    fig.suptitle('RL Progress - Parallel Games')
    
    text = 'learn: %0.4f \n' \
           'reg  : %0.4f \n' \
           'mom  : %0.2f \n' \
           'batch: %i \n' \
            % (L_RATE, REG_RATE, MOM_RATE, BATCH_SIZE)

    ax.text(0.98, 0, text,
            transform = ax.transAxes, fontsize = 9,
            verticalalignment = 'bottom',
            horizontalalignment = 'right')
    
    plt.show(block = False)
    plt.savefig('test_saves/plots/progress_%i_%i_para.pdf' %(BOARD_SIZE, N_PITS))
    input('PE.')
    plt.close()
            
        
        
        
    
    
    
    
    
    
    
    
    
    
        
    
    

        
    