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

from test2d import QPlayer as QP_Base, play_cgame


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
         [2, 2,  2, 2, 2, 2, 2, 2, 2,  2, 2]
         ]

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
                      [2, 2,  2, 2, 2, 2, 2, 2, 2,  2, 2]
                      ]
        
        self.n_pieces = 33
        
        if do_first_turn not in ('no', 'fixed', 'rand'):
            raise ValueError('Argument `do_first_turn` must be one of "no", "fixed" or "rand".')
        self.do_first_turn = do_first_turn
        
        if self.do_first_turn == 'fixed':
            self.board[4][4] = FREE
            self.n_pieces -= 1
        elif self.do_first_turn == 'rand':
            move = random.choice(((4, 4), (4, 6), (6, 4), (6, 6)))
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


## (Q-)Player
## ~~~~~~~~~~

class QPlayer(QP_Base):
    
    def __init__(self, *args):
        
        super().__init__(*args)
        self.trains = 0
    
    def get_action(self, state, at_random, legal_moves):
        
        lm_inds = np.where(legal_moves)[0]
        
        if at_random:
            ret = random.choice(lm_inds)
            #print(' - Random choice from', np.where(legal_moves)[0], 'was', ret)
            return ret
        
        else:
            ret = self.network.predict(np.array([state]))
            #print(' - Find max of', ret, 'in inds', lm_inds, end = '\n   -> ')
            ret = ret[0, lm_inds]
            #print(ret, ':', ret.argmax() : lm_inds[ret.argmax()])
            return lm_inds[ret.argmax()]
    
    def train(self, batch_size, l_rate, reg_rate, mom_rate):
        
        self.trains += 1
        
        # Get batch
        if batch_size < len(self.memory):
            batch = np.array(random.sample(self.memory, batch_size))
        else:
            batch = self.memory
        
        # Build examples and targets
        examples = []
        targets = []
        for state, action, reward, new_state, crashed, legal_moves in batch:
            
            # Build target
            target = self.network.predict(np.array([state]))[0] 
            target *= legal_moves
            
            target[action] = reward

            if not crashed:
                target[action] += self.disc_rate * max(self.network.predict(np.array([new_state]))[0])

            # Append to lists
            examples.append(state)
            targets.append(target)
            
        # Train
        self.network.train(np.array(examples), np.array(targets),
                           l_rate, 1, reg_rate, 1, mom_rate)



## Play game (human)
## ~~~~~~~~~

def print_board(game):
    
    st = '\n\n       Remain One\n\n'

    pi = 0
    for ri, row in enumerate(game.board):
        st += '  abcdefg  '[ri] + ' '
        for slot in row:
            st += ' '
            if slot == WALL:
                st += ' '
            else:
                st += 'o' if slot == FULL else '.'
                pi += 1
        st += '\n'

    st += '\n       1 2 3 4 5 6 7     \n\n'

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
    
    
    

## Play game (computer)
## ~~~~~~~~~

def play_cgame(game, player, epsilon, rand_start = False):
    
    game.restart(rand_start)
    
    hist = ''
    turns = 0
    new_state = game.get_state()
    while not (game.crashed or game.won):
        
        #print('Turn', turns, 'w lms', game.lms)
        #print_board(game)
        
        # Play turn
        state = new_state
        lms = game.lms
        action = player.get_action(state, 
                                   random.random() < epsilon,
                                   lms)
        reward = game.turn(action)
        new_state = game.get_state()

        turns += 1
        
        # Store in memory
        player.store(state, action, reward, new_state, game.crashed, lms)
        
        # History
        hist += str(action) + ' '
    
    return turns, game.won, hist


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

    SHAPE = [33, 
             400, 400, 
             len(Game._moves)]

    DISC_RATE = 0.95
    MAX_MEM_LEN = 1000

    BATCH_SIZE = 70
    L_RATE = 0.02
    REG_RATE = 0.0001
    MOM_RATE = 0.8

    NUM_EPOCHS = 1000
    EPSILON = lambda i: 1.3 - i/NUM_EPOCHS

    VERBOSE = 10
    DET_VERBOSE = 20
    
    
    game = Game()
    
    
    # Play game - human
    # ~~~~~~~~~~~~~~~~~
    
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
    
    player = QPlayer(MAX_MEM_LEN, DISC_RATE, SHAPE)
    
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
        player.train(BATCH_SIZE, L_RATE, REG_RATE, MOM_RATE)
        
        # If right epoch, play deterministic game
        if (epoch+1) % DET_VERBOSE == 0:
            
            #print('\n\n\n\n\n\n\n\nDET GAME\n\n\n\n\n\n\n\n\n\n')
            
            turns, won, this_hist = play_cgame(game, player, -1, False)
            
            print('\n  Played a deterministic game %i minutes into training.\n  Lasted %i turns and %s' \
                  %((time.time() - start_time) / 60,
                    turns,
                    'won!' if won else 'lost...')
                 )
            
            print('    Hist:', this_hist)
            if this_hist == last_hist:
                print('           PLAYED THE SAME GAME!!!')
            last_hist = this_hist
            
            det_turnlist.append(32 - turns)
            det_winlist.append(won)
            
            #input('PE.')
    
    print('\nDone running RL! Saving...')
    
    # All done, save network and turn list and stuff
    player.network.save('testRO_saves/network.json')
    
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
                       'NUM_EPOCHS': NUM_EPOCHS,
                       'DET_VERBOSE': DET_VERBOSE}
            }
    
    file = open('testRO_saves/results.json', 'w')
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
    plt.savefig('testRO_saves/progress.pdf')
    input('PE.')
    plt.close()
            
        
        
        
    
    
    
    
    
    
    
    
    
    
        
    
    

        
    