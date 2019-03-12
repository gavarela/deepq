## ~~~~~~~~~~~~~~~
#  Remain One Game
## ~~~~~~~~~~~~~~~

import os, numpy as np


## Variables
## ~~~~~~~~~

FULL = 1    # State of places/holes
FREE = 0
WALL = 2

UP    = (-1,  0)    # Directions
DOWN  = ( 1,  0)
LEFT  = ( 0, -1)
RIGHT = ( 0,  1)
DIRS = (RIGHT, LEFT, UP, DOWN)

BOARD = [[2, 2,  2, 2, 2, 2,  2, 2],
         [2, 2,  2, 2, 2, 2,  2, 2],

         [2, 2,  2, 1, 1, 2,  2, 2],
         [2, 2,  1, 1, 1, 1,  2, 2],
         [2, 2,  1, 1, 1, 1,  2, 2],
         [2, 2,  2, 1, 1, 2,  2, 2],
         
         [2, 2,  2, 2, 2, 2,  2, 2],
         [2, 2,  2, 2, 2, 2,  2, 2]
         ]

ROW_RANGE = range(2, 5+1)
COL_RANGE = {2: range(3, 4+1),
             3: range(2, 5+1),
             4: range(2, 5+1),
             5: range(3, 4+1)}

VALID   = True  # Validity of moves
INVALID = False


## Game Class
## ~~~~~~~~~~

class Game(object):
    
    def __init__(self):
        ''' Initialise board. '''
        
        self.board = [[2, 2,  2, 2, 2, 2,  2, 2],
                      [2, 2,  2, 2, 2, 2,  2, 2],

                      [2, 2,  2, 0, 1, 2,  2, 2],
                      [2, 2,  1, 1, 1, 1,  2, 2],
                      [2, 2,  1, 1, 1, 1,  2, 2],
                      [2, 2,  2, 1, 1, 2,  2, 2],

                      [2, 2,  2, 2, 2, 2,  2, 2],
                      [2, 2,  2, 2, 2, 2,  2, 2]
                      ]
        self.count()
        
        self.history = []
        self.done = False
        
    def count(self):
        ''' Count number of full places in board. '''
        
        self.num_pieces = 0
        for r in ROW_RANGE:
            for c in COL_RANGE[r]:
                if self.board[r][c] == FULL: self.num_pieces += 1
    
    def get_num_pieces(self):
        
        return self.num_pieces
        
    def is_done(self):
        ''' True if there's only one piece left or if there are no possible moves. '''
        
        if self.done: return self.done
        
        if self.num_pieces == 1:
            self.done = True
            return self.done
        
        # Test moves
        self.done = True
        for r in ROW_RANGE:
            for c in COL_RANGE[r]:

                if self.board[r][c] == FULL:
                    for direc in DIRS:
                        
                        if self.move((r, c), direc) == VALID:
                            self.done = False
        
        return self.done
    
    def move(self, start_ind, direc):
        ''' Tests for validity of moving piece in place indexed by \start_ind in the direction \direc. '''
        
        start_place = self.board[start_ind[0]][start_ind[1]]
        if start_place != FULL: return INVALID
        
        mid_place = self.board[start_ind[0] + direc[0]][start_ind[1] + direc[1]]
        if mid_place != FULL: return INVALID

        end_place = self.board[start_ind[0] + 2*direc[0]][start_ind[1] + 2*direc[1]]
        if end_place != FREE: return INVALID

        return VALID
    
    def do_move(self, start_ind, direc):
        ''' Empties starting and middle place and fills end place. Does NOT test whether move is valid. '''
        
        self.board[start_ind[0]][start_ind[1]] = FREE
        self.board[start_ind[0] + direc[0]][start_ind[1] + direc[1]] = FREE
        self.board[start_ind[0] + 2*direc[0]][start_ind[1] + 2*direc[1]] = FULL
        
    def turn(self, start_ind, direc):
        ''' Does move if move is valid. Returns reward 1/(self.num_pieces). '''
        
        if self.move(start_ind, direc) == VALID:
            self.do_move(start_ind, direc)
            self.num_pieces -= 1
            self.history.append((start_ind, direc))
            return 1/self.num_pieces
        
        if self.is_done():
            return -1
        else:
            return 0
    
    def first_turn(self, start_ind):
        ''' Does nothing because 1st piece is already removed from board (because only one possible first piece to remove to make the game work). '''
        
        return 0
    
    def __str__(self):

        ret = '\n\n       Remain One\n\n'

        pi = 0
        for ri, row in enumerate(self.board):
            ret += '  abcd  '[ri] + ' '
            for slot in row:
                ret += ' '
                if slot == WALL:
                    ret += ' '
                else:
                    ret += 'o' if slot == FULL else '.'
                    pi += 1
            ret += '\n'

        ret += '\n       1 2 3 4     \n\n'

        return ret
    
    def play(self):
        ''' Plays game. Re-initialises if was already played. '''
        
        if len(self.history) > 0:
            self.__init__()
        
        # First turn
        os.system('clear')
        print(self)
        
        r, c = 0, 0
        while r not in ROW_RANGE or c not in COL_RANGE[r]:
            inpt = input('\nChoose a piece to remove (e.g. "d4"): ')
            if len(inpt) < 2: inpt = '       ' 
            r = '  abcdefg'.find(inpt[0])
            c = int(inpt[1])+1 if inpt[1].isdigit() else 90
        
        self.first_turn((r, c))
        
        # Subsequent turns
        while not self.is_done():
            
            os.system('clear')
            print(self)
            
            r, c, direc = 0, 0, UP
            while self.move((r, c), direc) != VALID:
                r = 90
                while r not in ROW_RANGE or c not in COL_RANGE[r] or \
                    direc not in (UP, DOWN, LEFT, RIGHT):
                    inpt = input('\nChoose move (e.g. "f4 up"): ')
                    if len(inpt) < 5: inpt = '       ' 
                    r = '  abcdefg'.find(inpt[0])
                    c = int(inpt[1])+1 if inpt[1].isdigit() else 90
                    direc = {'up': UP, 'down': DOWN, 
                             'left': LEFT, 'right': RIGHT}.get(inpt[3:], 0)
            
            self.turn((r, c), direc)
        
        # End
        os.system('clear')
        print(self)
        
        print('Pieces left:', self.num_pieces)
        if self.num_pieces == 1:
            print('\n   Congratulations!\n')
        else:
            print('\n   Shame... Try again!\n')
        
        print('\nHistory:\n')
        item = self.history[0]
        print(' - Removed ' + '  abcd'[item[0]] + str(item[1]-1))
        for item in self.history[1:]:
            print(' - Moved '+ '  abcd'[item[0][0]] + str(item[0][1]-1) + ' ' + {UP: 'up', DOWN: 'down', LEFT: 'left', RIGHT: 'right'}[item[1]])
        print()
        

## Computer Game
## ~~~~~~~~~~~~~

class ComputerGame(Game):
    ''' Add list of valid moves. '''
    
    _computer_moves = []
    for r in ROW_RANGE:
        for c in COL_RANGE[r]:
            
            if BOARD[r][c] != WALL:
                for direc in DIRS:
                    
                    if BOARD[r + 2*direc[0]][c + 2*direc[1]] != WALL:
                        _computer_moves.append(((r, c), direc))
    
#    _computer_first_moves = []
#    for r in ROW_RANGE:
#        for c in ROW_RANGE:
#            if c in COL_RANGE[r]:
#                _computer_first_moves.append((r, c))
    
    def first_turn(self, move_ind):
        ''' Removes piece in _computer_moves indexed by \move_ind. '''
        
        ind = ComputerGame._computer_moves[move_ind][0]
        return super(ComputerGame, self).first_turn(ind)
    
    def turn(self, move_ind):
        ''' Does move indexed in _computer_moves by \move_ind. '''
        
        move = ComputerGame._computer_moves[move_ind]
        return super(ComputerGame, self).turn(move[0], move[1])
        
    def get_state(self, twod = True, for_keras = True):
        ''' If twod (2D) is False, returns 1d numpy array containing 0s for free places and 1s for full ones.
            If twod is True, returns the board (with WALLs -> FULLs) stripped of fully-walled rows and columns as a numpy array. 
            for_keras decides if shape is (7, 7, 1) or (1, 7, 7).
            '''
        
        ret = []
        if not twod:
            
            for r in ROW_RANGE:
                for c in COL_RANGE[r]:
                    if self.board[r][c] == FULL:
                        ret.append(1)
                    elif self.board[r][c] == FREE:
                        ret.append(0)
        
        else:
            
            for r in ROW_RANGE:
                ret.append([])
                
                for c in ROW_RANGE:
                    if self.board[r][c] == FULL:
                        ret[-1].append(1)
                    else:
                        ret[-1].append(0)
        
        ret = np.array(ret)
        
        if twod and for_keras:
            return ret.reshape(tuple(list(ret.shape) + [1]))
        elif twod:
            return ret.reshape(tuple([1] + list(ret.shape)))
        else:
            return ret
        
    def legal_moves(self, zero_ones = True):
        ''' Get array of legal move indices. If \zero_ones is True, return list of 0s or 1s, 1s corresponding to legal moves. '''
        
        legal_moves = []
        for move_ind, test_move in enumerate(ComputerGame._computer_moves):
            legal_moves.append(int(self.move(test_move[0], test_move[1]) == VALID))
        
        legal_moves = np.array(legal_moves)
        if not zero_ones: 
            legal_moves = np.where(legal_moves == 1)[0]
        
        return legal_moves
    
    @classmethod
    def get_inp_out_dims(cls, twod = True, for_keras = True):
        
        if not twod:
            inp_dims = sum(1 for r in ROW_RANGE for c in COL_RANGE[r])
        else:
            inp_dims = (len(ROW_RANGE), max([len(CR) for _, CR in COL_RANGE.items()]))
            if for_keras:
                inp_dims = tuple(list(inp_dims) + [1])
            elif twod:
                inp_dims = tuple([1] + list(inp_dims))
        
        return inp_dims, len(cls._computer_moves)
    
    def undo_move(self, start_ind, direc):
        ''' Fills starting and middle place and empties end place. Does NOT test whether move can be undone. '''
        
        self.board[start_ind[0]][start_ind[1]] = FULL
        self.board[start_ind[0] + direc[0]][start_ind[1] + direc[1]] = FULL
        self.board[start_ind[0] + 2*direc[0]][start_ind[1] + 2*direc[1]] = FREE
    
    def next_state(self, move_ind,
                   twod = True, for_keras = True):
        ''' Does move, store new state, undoes move, returns stored state. '''
        
        #print('Trying move ind:', move_ind)
        move = ComputerGame._computer_moves[move_ind]
        
        self.do_move(move[0], move[1])
        new_state = self.get_state(twod, for_keras)
        self.undo_move(move[0], move[1])
        
        return new_state
    

## Play
## ~~~~

if __name__ == "__main__":
    
    Game().play()
    
    
