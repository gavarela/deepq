## ~~~~~~~~~~~~~~~~
#  Tic Tac Toe Game
## ~~~~~~~~~~~~~~~~


## Imports and definitions
## ~~~~~~~~~~~~~~~~~~~~~~~


import random, os, numpy as np

EMPTY = 0
O = 1
X = 2

PIECE_DIC = {EMPTY: ' ', X: 'X', O: 'O'}


## Game
## ~~~~

class Game(object):
    
    _letters = ('a', 'b', 'c')
    _numbers = ('1', '2', '3')
    _header = '\n\n    Tic Tac Toe\n\n'
    _letter_dic = {'a': 0, 'b': 1, 'c': 2}
    
    def __init__(self):
        
        self.grid = np.zeros((3, 3), dtype = int)
        
        self.is_crashed = False
        self.current_player = X
        
    def check_crash(self):
        ''' Check if all places are full. '''
        
        if self.is_crashed: return True
        self.is_crashed = True
        
        for i in self.grid.ravel():
            if i == EMPTY: self.is_crashed = False
        
        return self.is_crashed
    
    def check_done(self):
        ''' Check if someone's won. Returns False or winning player (X or O). '''
        
        # Check vertical and horizontal lines
        for i in range(3):
            
            hline = ''.join(self.grid[i, :].astype(str))
            if hline in ('111', '222'): return int(hline[0])
            
            vline = ''.join(self.grid[:, i].astype(str))
            if vline in ('111', '222'): return int(vline[0])
            
        # Check diagonals
        dline = ''.join(self.grid[range(3), range(3)].astype(str))
        if dline in ('111', '222'): return int(dline[0])
        
        dline = ''.join(self.grid[range(3), range(-1, -4, -1)].astype(str))
        if dline in ('111', '222'): return int(dline[0])
        
        return False
    
    def switch_player(self):
        self.current_player = X if self.current_player == O else O
    
    def turn(self, ind):
        ''' Does turn if OK - places piece and inverts player. Returns 1 if won, 0 otherwise. '''
        
        if self.grid[ind[0], ind[1]] == EMPTY:
            self.grid[ind[0], ind[1]] = self.current_player
            self.switch_player()
        
        return int(self.check_done() is not False)
    
    def first_turn(self, ind): self.turn(ind)
    
    @staticmethod
    def clear_term():
        
        _ = os.system('clear')
    
    def __str__(self):
        
        ret = Game._header
        
        for ri, row in enumerate(self.grid):
            
            rowstr = Game._letters[ri] + ' '
            for si, slot in enumerate(row):
                rowstr += PIECE_DIC[slot] + '|'
                
            rowstr = rowstr[:-1] + '\n'
                
            if ri != 3-1:
                rowstr += '  ––––– \n'
                
            ret += rowstr
        
        ret += '\n  1 2 3  \n\n'
        
        return ret
    
    def play(self):
        ''' Play (human) game. '''
        
        self.clear_term()
        print(self)
        
        # Turn loop
        while (not self.check_crash()) and (not self.check_done()):
            
            inpt = ''
            loop = True
            while loop:
                
                print('\n')
                inpt = input('Choose a place to put your piece (' + PIECE_DIC[self.current_player] + '): ')
                
                if len(inpt) > 1:
                    if (inpt[0] in Game._letters and \
                        inpt[1] in Game._numbers):
                        loop = False
            
            self.turn((Game._letter_dic[inpt[0]], int(inpt[1]) - 1))
            
            self.clear_term()
            print(self)
            
        # Finish
        done = self.check_done()
        print('\n\n')
        if done is not False:
            print('Congratulations! ' + PIECE_DIC[done] + ' won!')
        else:
            print('Shame! Nobody wins...')
    

## Computer Game
## ~~~~~~~~~~~~~

class ComputerGame(Game):
    
    def __init__(self):
        super().__init__()
        self.done = False
        
    def turn(self, move_ind):
        ''' Adds piece to (ind//3, ind%3). '''
        
        return super().turn((move_ind//3, move_ind%3))
    
    def get_state(self, twod = True, for_keras = False):
        ''' Returns with 1s where the current player is and 2s where the opponent is. '''

        grid = self.grid.ravel()
        
        if self.current_player != O:
            grid = np.array([i if i == EMPTY else 1 if i == X else 2 for i in grid])
        
        if twod:
            if for_keras:
                grid = grid.reshape((3, 3, 1))
            else:
                grid = grid.reshape((1, 3, 3))
        
        return grid
    
    def legal_moves(self, zero_ones = True):
        
        where = np.where(self.grid.ravel() == EMPTY)[0]
        if not zero_ones: return where
        
        return np.array([0 if i not in where else 1 for i in range(9)])
    
    @classmethod
    def get_inp_out_dims(cls, twod = True, for_keras = False):
        
        if not twod:
            inp_dims = 9
        else:
            if for_keras:
                inp_dims = (3, 3, 1)
            elif twod:
                inp_dims = (1, 3, 3)
        
        return inp_dims, 9
    
    def undo_move(self, ind):
        
        self.grid[ind[0], ind[1]] = EMPTY
        self.switch_player()
    
    def next_state(self, move_ind,
                   twod = True, for_keras = False):
        ''' Does move, store new state, undoes move, returns stored state. '''
        
        #print('Trying move ind:', move_ind)
        move = (move_ind//3, move_ind%3)
        
        self.turn(move)
        new_state = self.get_state(twod, for_keras)
        self.undo_move(move)
        
        return new_state
    
    def is_done(self):
        
        self.done = self.check_crash() or self.check_done()
        return self.done
        

if __name__ == "__main__":
    
    Game().play()
            
                    
                    
            
            
            