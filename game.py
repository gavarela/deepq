## ~~~~~~~~~~~~~~~~~~~~
#  All Remain One games
## ~~~~~~~~~~~~~~~~~~~~

'''

Maybe eventually we'll be able to have one single Game class and we just have to define the board and possible starting positions and row_range, etc. in each new file for new board sizes.

'''

import random, numpy as np, os


## BOARD_SIZES
## ~~~~~~~~~~~

_BOARDs = [
    
    {'BOARD': [[2, 1, 1, 2],
               [1, 1, 1, 1],
               [1, 1, 1, 1],
               [2, 1, 1, 2]],
    'DET_START': (0, 1),
    'ALT_STARTS': ((0, 1), (0, 2),
                   (1, 0), (2, 0),
                   (3, 1), (3, 2),
                   (1, 3), (2, 3))},
    
    {'BOARD': [[2, 1, 1, 1, 2],
               [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1],
               [2, 1, 1, 1, 2]],
    'DET_START': (0, 1),
    'ALT_STARTS': ((0, 1), (0, 3),
                   (3, 1), (3, 3),
                   (1, 2), (2, 2))},
    
    {'BOARD': [[2, 1, 1, 1, 2],
               [2, 1, 1, 1, 2],
               [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1],
               [2, 1, 1, 1, 2],
               [2, 1, 1, 1, 2]],
    'DET_START': (2, 2),
    'ALT_STARTS': ((2, 2), (3, 2),
                   (0, 2), (5, 2))},
    
    {'BOARD': [[2, 1, 1, 1, 2],
               [2, 1, 1, 1, 2],
               [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1],
               [2, 1, 1, 1, 2],
               [2, 1, 1, 1, 2]],
    'DET_START': (3, 1),
    'ALT_STARTS': ((3, 1), (3, 3), (2, 2), (4, 2))},
    
    {'BOARD': [[2, 2, 1, 1, 1, 2, 2],
               [2, 2, 1, 1, 1, 2, 2],
               [1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1],
               [2, 2, 1, 1, 1, 2, 2],
               [2, 2, 1, 1, 1, 2, 2]],
    'DET_START': (3, 3),
    'ALT_STARTS': ((3, 3),
                   (2, 2), (2, 4), (4, 2), (4, 4))}
    
]


## Full RO
## ~~~~~~~

class Game(object):
    
    # State of places/holes
    _FULL = 1
    _FREE = 0
    _WALL = 2
    
    # Directions
    _UP    = (-1,  0)
    _DOWN  = ( 1,  0)
    _LEFT  = ( 0, -1)
    _RIGHT = ( 0,  1)
    _DIRS = (_RIGHT, _LEFT, _UP, _DOWN)
    
    # Validity of moves
    _VALID   = True
    _INVALID = False
        
    def __init__(self, board_size = -1, do_first_turn = 'fixed'):
        
        if do_first_turn not in ('no', 'fixed', 'rand'):
            raise ValueError('Argument `do_first_turn` must be one of "no", "fixed" or "rand".')
        self.do_first_turn = do_first_turn
        
        self.init_class(board_size)
        self.init_game(do_first_turn)
    
    def init_class(self, board_size):
        
        # Get board and initial moves
        self._BOARD = _BOARDs[board_size]['BOARD']
        self._BOARD = [[2, 2] + row + [2, 2] for row in self._BOARD]
        self._BOARD = [[2 for c in range(len(self._BOARD[0]))]]*2 + \
                        self._BOARD + \
                        [[2 for c in range(len(self._BOARD[0]))]]*2
        
        self._TOT_PIECES = sum([p == 1 \
                                for ROW in self._BOARD \
                                for p in ROW])
        
        self._DET_START  = _BOARDs[board_size]['DET_START']
        self._ALT_STARTS = _BOARDs[board_size]['ALT_STARTS']
        
        # Get row and column range of playable board
        self._ROW_RANGE = [r for r, row in enumerate(self._BOARD) \
                           if sum([place != self._WALL for place in row]) != 0]
        self._COL_RANGE = {r: [c for c, p in enumerate(self._BOARD[r]) \
                               if p != self._WALL] \
                           for r in self._ROW_RANGE}
        
        # Get list of valid moves
        self._MOVES = []
        for r in self._ROW_RANGE:
            for c in self._COL_RANGE[r]:

                if self._BOARD[r][c] != self._WALL:
                    for direc in self._DIRS:

                        if self._BOARD[r + 2*direc[0]][c + 2*direc[1]] != self._WALL:
                            self._MOVES.append(((r, c), direc))
        
        # Get number of places
        self._N_PLACES = sum([1 for row in self._BOARD \
                              for p in row \
                              if p == self._FULL])
    
    def init_game(self, do_first_turn):
        
        # Initialise board
        self.board = [row.copy() for row in self._BOARD]
        self.n_pieces = self._N_PLACES
        
        if self.do_first_turn == 'fixed':
            self.board[2+self._DET_START[0]][2+self._DET_START[1]] = self._FREE
            self.n_pieces -= 1
        elif self.do_first_turn == 'rand':
            start = random.choice(self._ALT_STARTS)
            self.board[2+start[0]][2+start[1]] = self._FREE
            self.n_pieces -= 1
        
        # Initialise self
        self.crashed = False
        self.won = False
        
        self.check_legal_moves()
        
    def check_move(self, start_ind, direc):
        
        start_place = self.board[start_ind[0]][start_ind[1]]
        if start_place != self._FULL: return self._INVALID
        
        mid_place = self.board[start_ind[0] + direc[0]][start_ind[1] + direc[1]]
        if mid_place != self._FULL: return self._INVALID

        end_place = self.board[start_ind[0] + 2*direc[0]][start_ind[1] + 2*direc[1]]
        if end_place != self._FREE: return self._INVALID

        return self._VALID
        
    def turn(self, move_ind):
        
        start_ind, direc = self._MOVES[move_ind]
        
        if self.check_move(start_ind, direc):
            
            self.board[start_ind[0]][start_ind[1]] = self._FREE
            self.board[start_ind[0] + direc[0]][start_ind[1] + direc[1]] = self._FREE
            self.board[start_ind[0] + 2*direc[0]][start_ind[1] + 2*direc[1]] = self._FULL
        
            self.n_pieces -= 1
            
            if self.n_pieces == 1:
                self.won = True
            
            self.check_legal_moves()
            return 1/self.n_pieces
        
        return 0
    
    def check_legal_moves(self):
        
        self.lms = []
        for move_ind, move in enumerate(self._MOVES):
            self.lms.append(int(self.check_move(move[0], move[1])))
        
        if sum(self.lms) == 0:
            self.crashed = True
        
    def get_state(self):
        
        state = []
        for r in self._ROW_RANGE:
            for c in self._COL_RANGE[r]:
                state.append(self.board[r][c])
        
        return state
    
    def restart(self, rand = False):
        
        if rand:
            self.init_game('rand')
        else:
            self.init_game(self.do_first_turn)
    
    def print_board(self):
    
        letters = '  ' + 'abcdefg'[:(len(self._BOARD)-4)] + '  '
        st = '\n\n       Remain One\n\n'

        p = 0
        for r, row in enumerate(self.board):
            st += letters[r] + ' '
            for slot in row:
                st += ' '
                if slot == self._WALL:
                    st += ' '
                else:
                    st += 'o' if slot == self._FULL else '.'
                    p += 1
            st += '\n'

        st += '\n       '
        for i in range(len(self.board[0])-4):
            st += str(i+1) + ' '
        st += '  \n\n'

        print(st)

    # For human to play
    def play(self):

        self.restart()
        
        print(self.crashed, self.won)
        
        while not (self.crashed or self.won):

            # Print board/state
            os.system('clear')
            self.print_board()

            # Get move from input
            r, c, direc = 90, 0, self._UP
            while r not in self._ROW_RANGE or \
                    c not in self._COL_RANGE[r] or \
                    direc not in self._DIRS:

                inpt = input('\nChoose move (e.g. "f4 up"): ')
                if len(inpt) < 5: inpt = '       '
                r = '  abcdefg'.find(inpt[0])
                c = int(inpt[1])+1 if inpt[1].isdigit() else 90
                direc = {'up':   self._UP,   'down':  self._DOWN, 
                         'left': self._LEFT, 'right': self._RIGHT}.get(inpt[3:], 0)

            # Do move
            try:
                move_ind = self._MOVES.index(((r, c), direc))
            except:
                continue

            print('\nMove with ind:', move_ind, 
                  '\nReward:', self.turn(move_ind))

        os.system('clear')
        self.print_board()

        print('Pieces left:', self.n_pieces)
        if game.won:
            print('\n   Congratulations!\n')
        else:
            print('\n   Shame... Try again!\n')


# Play game (human)
# ~~~~~~~~~

if __name__ == "__main__":
    
    import sys
    
    game = Game(int(sys.argv[1]) if len(sys.argv) == 2 else -1)
        
    print('Welcome!')
    another = True
    while another:
        game.play()
        another = input('Enter anything to play again, nothing to quit. ') != ''