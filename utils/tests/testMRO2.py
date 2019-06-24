## ~~~~~~~~~~~~~~~~~~
#  Testing Q-Learning
#  Small Remain One
## ~~~~~~~~~~~~~~~~~~


'''

IF WANT TO CHANGE BOARD SIZE, FOLLOWING CHANGES NEED TO BE MADE:
 - BOARD, N_PIECES, ROW_RANGE and COL_RANGE
 - In Game __init__: self.board, self.n_pieces and self.do_first_move stuff
 - In print_board: Indexing (abc and 123)
 - In det game bit: Score appended to det_turnlist
 - SHAPE of network

'''

import sys
if "../" not in sys.path:
    sys.path.append("../")
    
import numpy as np, random, json, os
from collections import deque

from WillyNet import WillyNet
from DW import *

from testSRO import QPlayer, play_cgame, play_hgame


## Game
## ~~~~

FULL = 1    # State of places/holes
FREE = 0
WALL = 2

UP    = (-1,  0)    # Directions
DOWN  = ( 1,  0)
LEFT  = ( 0, -1)
RIGHT = ( 0,  1)
DIRS = (RIGHT, LEFT, UP, DOWN)

BOARD = [[2, 2,  2, 2, 2, 2, 2,  2, 2],
         [2, 2,  2, 2, 2, 2, 2,  2, 2],

         [2, 2,  2, 1, 1, 1, 2,  2, 2],
         [2, 2,  2, 1, 1, 1, 2,  2, 2],
         [2, 2,  1, 1, 1, 1, 1,  2, 2],
         [2, 2,  1, 1, 1, 1, 1,  2, 2],
         [2, 2,  2, 1, 1, 1, 2,  2, 2],
         [2, 2,  2, 1, 1, 1, 2,  2, 2],

         [2, 2,  2, 2, 2, 2, 2,  2, 2],
         [2, 2,  2, 2, 2, 2, 2,  2, 2]
         ]

N_PIECES = 22

ROW_RANGE = range(2, 7+1)
COL_RANGE = {2: range(3, 5+1),
             3: range(3, 5+1),
             4: range(2, 6+1),
             5: range(2, 6+1),
             6: range(3, 5+1),
             7: range(3, 5+1)}

VALID   = True  # Validity of moves
INVALID = False

class Game(object):
    
    _moves = []
    for r in ROW_RANGE:
        for c in COL_RANGE[r]:
            
            if BOARD[r][c] != WALL:
                for direc in DIRS:
                    
                    if BOARD[r + 2*direc[0]][c + 2*direc[1]] != WALL:
                        _moves.append(((r, c), direc))
    
    def __init__(self, do_first_turn = 'fixed'):
        
        self.board = [[2, 2,  2, 2, 2, 2, 2,  2, 2],
                      [2, 2,  2, 2, 2, 2, 2,  2, 2],
 
                      [2, 2,  2, 1, 1, 1, 2,  2, 2],
                      [2, 2,  2, 1, 1, 1, 2,  2, 2],
                      [2, 2,  1, 1, 1, 1, 1,  2, 2],
                      [2, 2,  1, 1, 1, 1, 1,  2, 2],
                      [2, 2,  2, 1, 1, 1, 2,  2, 2],
                      [2, 2,  2, 1, 1, 1, 2,  2, 2],
 
                      [2, 2,  2, 2, 2, 2, 2,  2, 2],
                      [2, 2,  2, 2, 2, 2, 2,  2, 2]
                      ]
        
        self.n_pieces = N_PIECES
        
        if do_first_turn not in ('no', 'fixed', 'rand'):
            raise ValueError('Argument `do_first_turn` must be one of "no", "fixed" or "rand".')
        self.do_first_turn = do_first_turn
        
        if self.do_first_turn == 'fixed':
            self.board[4][4] = FREE
            self.n_pieces -= 1
        elif self.do_first_turn == 'rand':
            move = random.choice(((4, 4), (5, 4))) #,
#                                  (2, 4), (7, 4)))
            self.board[move[0]][move[1]] = FREE
            self.n_pieces -= 1
        
        self.crashed = False
        self.won = False
        
        self.check_legal_moves()
        
    def check_move(self, start_ind, direc):
        
        start_place = self.board[start_ind[0]][start_ind[1]]
        if start_place != FULL: return INVALID
        
        mid_place = self.board[start_ind[0] + direc[0]][start_ind[1] + direc[1]]
        if mid_place != FULL: return INVALID

        end_place = self.board[start_ind[0] + 2*direc[0]][start_ind[1] + 2*direc[1]]
        if end_place != FREE: return INVALID

        return VALID
        
    def turn(self, move_ind):
        
        start_ind, direc = Game._moves[move_ind]
        
        if self.check_move(start_ind, direc):
            
            self.board[start_ind[0]][start_ind[1]] = FREE
            self.board[start_ind[0] + direc[0]][start_ind[1] + direc[1]] = FREE
            self.board[start_ind[0] + 2*direc[0]][start_ind[1] + 2*direc[1]] = FULL
        
            self.n_pieces -= 1
            
            if self.n_pieces == 1:
                self.won = True
            
            self.check_legal_moves()
            return 1/self.n_pieces
        
        return 0
    
    def check_legal_moves(self):
        
        self.lms = []
        for move_ind, move in enumerate(Game._moves):
            self.lms.append(int(self.check_move(move[0], move[1])))
        
        if sum(self.lms) == 0:
            self.crashed = True
        
    def get_state(self):
        
        state = []
        for r in ROW_RANGE:
            for c in COL_RANGE[r]:
                state.append(self.board[r][c])
        
        return state
    
    def restart(self, rand = False):
        
        if rand:
            self.__init__('rand')
        else:
            self.__init__(self.do_first_turn)


## Convolutional QPlayer
## ~~~~~~~~~~~~~~~~~~~~~

class ConvQPlayer(QPlayer):
    
    def __init__(self, max_mem_len, disc_rate, shape):
        
        self.max_mem_len = max_mem_len
        self.disc_rate = disc_rate
        
        self.memory = deque()
        
        self.network = DeepWilly('quadratic')
        for willy in willies:
            self.network.add(willy['class'](*willy['args']))
        



## Print board (for hgame)
## ~~~~~~~~~~~

def print_board(game):
    
    st = '\n\n       Remain One\n\n'

    pi = 0
    for ri, row in enumerate(game.board):
        st += '  abcdef  '[ri] + ' '
        for slot in row:
            st += ' '
            if slot == WALL:
                st += ' '
            else:
                st += 'o' if slot == FULL else '.'
                pi += 1
        st += '\n'

    st += '\n       1 2 3 4 5     \n\n'

    print(st)

def play_hgame(game):
    
    game.restart()
    
    while not (game.crashed or game.won):
        
        # Print board/state
        #os.system('clear')
        print_board(game)
        
        print('State is:', game.get_state())
        #print('Legal moves are:', np.where(game.lms)[0])
        
        # Get move from input
        r, c, direc = 90, 0, UP
        while r not in ROW_RANGE or \
                c not in COL_RANGE[r] or \
                direc not in (UP, DOWN, LEFT, RIGHT):
            
            inpt = input('\nChoose move (e.g. "f4 up"): ')
            if len(inpt) < 5: inpt = '       '
            r = '  abcdefg'.find(inpt[0])
            c = int(inpt[1])+1 if inpt[1].isdigit() else 90
            direc = {'up': UP, 'down': DOWN, 
                     'left': LEFT, 'right': RIGHT}.get(inpt[3:], 0)

        # Do move
        try:
            move_ind = Game._moves.index(((r, c), direc))
        except:
            continue
        
        print('\nMove with ind:', move_ind, 
              '\nReward:', game.turn(move_ind))
        
    #os.system('clear')
    print_board(game)
    
    print('Pieces left:', game.n_pieces)
    if game.won:
        print('\n   Congratulations!\n')
    else:
        print('\n   Shame... Try again!\n')



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
    
    PLAYER_CLASS = QPlayer
    SHAPE = [N_PIECES, 
             600, 400, 
             len(Game._moves)]

    DISC_RATE = 0.95
    MAX_MEM_LEN = 1000

    MEM_BATCH = 150
    TRAIN_BATCH = 1/5
    
    L_RATE = 0.02 * MEM_BATCH * TRAIN_BATCH
    REG_RATE = 0.0001
    MOM_RATE = 0.8

    NUM_EPOCHS = 30000
    EPSILON = lambda i: 1.3 - i/NUM_EPOCHS

    VERBOSE = 25
    DET_VERBOSE = 50
    RAND_DETS = False
    
    SAVE_DIR = 'testMRO_saves/MRO2/batches/e%i_m%i_t%i_l%s' %(NUM_EPOCHS, MEM_BATCH, TRAIN_BATCH * MEM_BATCH, str(L_RATE).replace('.', '-'))
    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)
        os.mkdir(SAVE_DIR + '/checkpoints')
    
    
    game = Game()
    
    
    # Play human game(s)
    # ~~~~~~~~~~~~~~~~~~
    
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
    
    player = PLAYER_CLASS(MAX_MEM_LEN, DISC_RATE, SHAPE)
    
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
            
            #print('\n\n\n\n\n\n\n\nDET GAME\n\n\n\n\n\n\n\n\n\n')
            
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
            
            det_turnlist.append(N_PIECES-1 - turns)
            det_winlist.append(won)
            
        # Every 1000 epochs, plot and save net
        if (epoch+1) % 1000 == 0:
            
            fig, ax = plt.subplots()

            for i in range(len(det_winlist)):
                ax.plot(i * DET_VERBOSE, det_turnlist[i], 
                        'g.' if det_winlist[i] else 'r.',
                        markersize = 1)

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Pieces Left')
            ax.set_title('RL Progress - Epoch %i' %(epoch+1))

            text = 'learn: %0.4f \n' \
                   'reg: %0.4f \n' \
                   'mom: %0.2f \n' \
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
            
            #try: plt.close()
            plt.show(block = False)
            plt.savefig(SAVE_DIR + '/progress_%i.pdf' %(epoch+1))
            
            player.network.save(SAVE_DIR + '/network_%i.json' %(epoch+1))
    
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
           'reg  : %0.4f \n' \
           'mom  : %0.2f \n' \
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
        
    
    
    
    
    
    
    
    
    
        
    
    

        
    