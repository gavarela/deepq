## ~~~~~~~~~~~~~~~~~~
#  Testing Q-Learning
#  Med Remain One 3
## ~~~~~~~~~~~~~~~~~~


'''

IF WANT TO CHANGE BOARD SIZE, FOLLOWING CHANGES NEED TO BE MADE:
 - BOARD, N_PIECES, ROW_RANGE and COL_RANGE
 - In Game __init__: self.board, self.n_pieces and self.do_first_move stuff
 - In print_board: Indexing (abc and 123)
 - In det game bit: Score appended to det_turnlist
 - SHAPE of network
 

MRO2 times:
playing game is ~ 0.002s
training is ~ 0.04s

'''

import sys
if "../" not in sys.path:
    sys.path.append("../")
    
import numpy as np, random, json, os
import matplotlib.pyplot as plt
from collections import deque

from WillyNet import WillyNet
from DW import *

from testSRO import play_cgame, print_board
from testMRO2_checkpoints import QPlayer, save_progress


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

BOARD = [[2, 2,  2, 2, 2, 2, 2, 2, 2,  2, 2],
         [2, 2,  2, 2, 2, 2, 2, 2, 2,  2, 2],
         
         [2, 2,  2, 2, 1, 1, 1, 2, 2,  2, 2],
         [2, 2,  2, 2, 1, 1, 1, 2, 2,  2, 2],
         [2, 2,  1, 1, 1, 1, 1, 1, 1,  2, 2],
         [2, 2,  1, 1, 1, 1, 1, 1, 1,  2, 2],
         [2, 2,  1, 1, 1, 1, 1, 1, 1,  2, 2],
         [2, 2,  2, 2, 1, 1, 1, 2, 2,  2, 2],
         [2, 2,  2, 2, 1, 1, 1, 2, 2,  2, 2],

         [2, 2,  2, 2, 2, 2, 2, 2, 2,  2, 2],
         [2, 2,  2, 2, 2, 2, 2, 2, 2,  2, 2],
         ]

N_PIECES = 33

ROW_RANGE = range(2, 8+1)
COL_RANGE = {2: range(4, 6+1),
             3: range(4, 6+1),
             4: range(2, 8+1),
             5: range(2, 8+1),
             6: range(2, 8+1),
             7: range(4, 6+1),
             8: range(4, 6+1)}

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
        
        self.board = [[2, 2,  2, 2, 2, 2, 2, 2, 2,  2, 2],
                      [2, 2,  2, 2, 2, 2, 2, 2, 2,  2, 2],
 
                      [2, 2,  2, 2, 1, 1, 1, 2, 2,  2, 2],
                      [2, 2,  2, 2, 1, 1, 1, 2, 2,  2, 2],
                      [2, 2,  1, 1, 1, 1, 1, 1, 1,  2, 2],
                      [2, 2,  1, 1, 1, 1, 1, 1, 1,  2, 2],
                      [2, 2,  1, 1, 1, 1, 1, 1, 1,  2, 2],
                      [2, 2,  2, 2, 1, 1, 1, 2, 2,  2, 2],
                      [2, 2,  2, 2, 1, 1, 1, 2, 2,  2, 2],
 
                      [2, 2,  2, 2, 2, 2, 2, 2, 2,  2, 2],
                      [2, 2,  2, 2, 2, 2, 2, 2, 2,  2, 2],
                      ]
        
        self.n_pieces = N_PIECES
        
        if do_first_turn not in ('no', 'fixed', 'rand'):
            raise ValueError('Argument `do_first_turn` must be one of "no", "fixed" or "rand".')
        self.do_first_turn = do_first_turn
        
        if self.do_first_turn == 'fixed':
            self.board[5][5] = FREE
            self.n_pieces -= 1
        elif self.do_first_turn == 'rand':
            move = random.choice(((5, 5),
                                  (4, 4), (4, 6), (6, 4), (6, 6)))
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


## Play Game (for humans)
## ~~~~~~~~~

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
    
    import time, matplotlib.pyplot as plt, re
    
    print('\033[33m')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('\n     Test - Simple RemainOne     \n')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('\033[0m')
    
    ## Params
    ## ~~~~~~
    
    PLAYER_CLASS = QPlayer
    LOAD_FROM = None #'testMRO_saves/RO/e500k g5 mb500 tb50 lr0-08 rr0-0001 mr0-8 norand'
    
    if LOAD_FROM is not None:
        
        SAVE_DIR = LOAD_FROM
        
        filenames = os.listdir(LOAD_FROM + '/checkpoints')
        matches = [[m for m in re.finditer('_(\d+)\.', file)] for file in filenames]
        numbers = [int(match[0].group(1)) if len(match) > 0 else -1 for match in matches]
        CURRENT_EPOCH = max(numbers)
        
        # Data file
        file = open(LOAD_FROM + '/checkpoints/results_%i.json' %CURRENT_EPOCH, 'r')
        data = json.load(file)
        file.close()
        
        PARAMS = data['params']
        
        turnlist = data['turnlist']
        winlist = data['winlist']
        
        det_turnlist = data['det_turnlist']
        det_winlist = data['det_winlist']
        
        time_base = data['time']
        
        # Network
        player = PLAYER_CLASS(PARAMS['MAX_MEM_LEN'],
                              PARAMS['DISC_RATE'],
                              LOAD_FROM + '/checkpoints/network_%i.json' %CURRENT_EPOCH)
        
    else:
        
        # Params
        PARAMS = {'DISC_RATE': 0.95,
                  'MAX_MEM_LEN': 3000,

                  'MEM_BATCH': 500,
                  'TRAIN_BATCH': 1/10,

                  'REG_RATE': 0.0001,
                  'MOM_RATE': 0.8,

                  'NUM_EPOCHS': 500000,
                  'GAMES_PER_EPOCH': 5,
                  
                  'DET_VERBOSE': 500,
                  'RAND_DETS': False}
        
        PARAMS['L_RATE'] = 0.08 # 0.02 * PARAMS['MEM_BATCH'] * PARAMS['TRAIN_BATCH']
        
        SAVE_DIR = 'testMRO_saves/RO/e%ik g%i mb%i tb%i lr%s rr%s mr%s %srand' \
            %(PARAMS['NUM_EPOCHS']/1e3,
              PARAMS['GAMES_PER_EPOCH'],
              PARAMS['MEM_BATCH'],
              PARAMS['TRAIN_BATCH'] * PARAMS['MEM_BATCH'],
              str(PARAMS['L_RATE']).replace('.', '-'),
              str(PARAMS['REG_RATE']).replace('.', '-'),
              str(PARAMS['MOM_RATE']).replace('.', '-'),
              '' if PARAMS['RAND_DETS'] else 'no')
    
        if not os.path.isdir(SAVE_DIR):
            os.mkdir(SAVE_DIR)
            os.mkdir(SAVE_DIR + '/checkpoints')
        
        # Player
        SHAPE = [N_PIECES, 
                 800, 1000, 800,
                 len(Game._moves)]
        player = PLAYER_CLASS(PARAMS['MAX_MEM_LEN'],
                              PARAMS['DISC_RATE'],
                              SHAPE)
        
        # For training
        turnlist, winlist = [], []
        det_turnlist, det_winlist = [], []
        time_base = 0
        
        CURRENT_EPOCH = 0
    
    # Final params
    EPSILON = lambda i: 1.3 - i/PARAMS['NUM_EPOCHS']
    
    SAVE_EVERY = 2500
    VERBOSE = PARAMS['NUM_EPOCHS']+1
    
    game = Game()
    
    
#    # Play human game(s)
#    # ~~~~~~~~~~~~~~~~~~
#    
#    play = input('Enter anything to play the game: ')
#    while play != '':
#        play_hgame(game)
#        play = input('Enter anything to play again: ')
    
    
    # Reinforcement learning
    # ~~~~~~~~~~~~~~~~~~~~~~
    
    print('\033[32m\n')
    print('Training via RL')
    print('~~~~~~~~~~~~~~~')
    print('\033[0m')
    
    print('Parameters:',
          '\n - discount rate       =', PARAMS['DISC_RATE'],
          '\n - max memory len      =', PARAMS['MAX_MEM_LEN'], '\n',
          '\n - memory batch size   =', PARAMS['MEM_BATCH'],
          '\n - training batch size =', PARAMS['TRAIN_BATCH'], '\n',
          '\n - learning rate       =', PARAMS['L_RATE'],
          '\n - regularisation rate =', PARAMS['REG_RATE'],
          '\n - momentum rate       =', PARAMS['MOM_RATE'], '\n',
          '\n - epochs              =', PARAMS['NUM_EPOCHS'], 
          '\n - epsilon(i)          = 1.3 - i/epochs', '\n')
    
    last_hist = ''
    
    start_time = time.time()
    for epoch in range(CURRENT_EPOCH, PARAMS['NUM_EPOCHS']):
        
        # Play a game
        for game_i in range(1 + (PARAMS['GAMES_PER_EPOCH']-1) * max(0, min(1, round(EPSILON(epoch))))):
            turns, won, _ = play_cgame(game, player,   EPSILON(epoch))
            
            turnlist.append(turns)
            winlist.append(game.won)
        
        if (epoch+1) % VERBOSE == 0:
            print('\nEpoch %i: played %i turn(s) and %s' %(epoch+1, turns, 'won!' if won else 'lost...'))
        
        # Train
        s2 = time.time()
        player.train(PARAMS['MEM_BATCH'],
                     PARAMS['TRAIN_BATCH'], 
                     PARAMS['L_RATE'],
                     PARAMS['REG_RATE'],
                     PARAMS['MOM_RATE'])
        
        # If right epoch, play deterministic game
        if (epoch+1) % PARAMS['DET_VERBOSE'] == 0:
            
            turns, won, this_hist = play_cgame(game, player, -1, PARAMS['RAND_DETS'])
            
            print('\n  Played a deterministic game %i minutes (%i epochs) into training.\n  Lasted %i turns and %s' \
                  %((time.time() - start_time) / 60,
                    epoch+1,
                    turns,
                    'won!' if won else 'lost...')
                 )
            
            print('    Hist:', this_hist)
            if this_hist == last_hist:
                print('           PLAYED THE SAME GAME!!!')
            last_hist = this_hist
            
            det_turnlist.append(N_PIECES-1 - turns)
            det_winlist.append(won)
            
        # Every `SAVE_EVERY` epochs, plot and save net
        if (epoch+1) % SAVE_EVERY == 0:
            
            save_progress(turnlist, winlist, 
                          det_turnlist, det_winlist,
                          PARAMS,
                          time_base + time.time() - start_time,
                          player,
                          SAVE_DIR+'/checkpoints', 
                          name_end = '_%i' %(epoch+1))
            plt.close()
            
    
    end_time = time.time()
    print('\nDone running RL! Saving...')
    
    # All done, save data and plot
    save_progress(turnlist, winlist, 
                  det_turnlist, det_winlist,
                  PARAMS,
                  time_base + end_time - start_time,
                  player,
                  SAVE_DIR)
    
    input('PE.')
    plt.close()
        
    
    
    
    
    
    
    
    
    
        
    
    

        
    