## ~~~~~~~~~~~~~~~
#  Remain One Game
## ~~~~~~~~~~~~~~~

''' Implements a simple game where you move pieces by 'eating' neighbouring pieces, jumping over them. Eaten pieces are removed from the game. The objective is to have only one piece left by the end (when you can't make any more moves). '''

## Third-party libraries
## ~~~~~~~~~~~~~~~~~~~~~

import os
import numpy as np


## Game class
## ~~~~~~~~~~~

class Game(object):
    
    _tandb = [None, None, 0, 0, 0, None, None]
    _mid = [0, 0, 0, 0, 0, 0, 0]
    _grid = np.array([_tandb, _tandb, _mid, _mid, _mid, _tandb, _tandb])
    
    def __init__(self, verbose = False):
        ''' Create and fill board array. '''
        
        self.verbose = verbose
        
        self.pieces = []
        self.places = []

        for i in range(7):
            for j in range(7):
                if Game._grid[i, j] is not None:
                    self.places.append(Place((i, j), self))
                    self.pieces.append(Piece(self, self.places[-1]))
        
        self.history = []
        self.crashed = False
        
    def get_place_ind(self, ind):
        ''' Given grid index, give index of self.places for place with that index. '''
        
        pind = (-1, -1)
        i = -1
        while pind != ind:
            i += 1
            pind = self.places[i].ind
            
        return i
    
    def get_place(self, ind):
        ''' Given grid index, give place in self.places with that index. '''
        
        return self.places[self.get_place_ind(ind)]
    
    def get_piece_ind(self, piece):
        ' Given piece, gives its index in self.pieces. '
        
        for pi, p in enumerate(self.pieces):
            if p == piece:
                return pi
        
        assert False, "Piece doesn't exist!"
            
    def crash(self):
        ''' Looks for valid moves by looping over pieces and seeing if there are valid moves from there. '''
        
        if self.crashed: return True
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        self.crashed = True
        
        for piece in self.pieces:
            for direction in directions:
                if Move(self, piece.ind, direction).is_valid():
                    self.crashed = False
        
        return self.crashed
    
    def force_crash(self):
        
        self.crashed = True
        
    @staticmethod
    def clear_term():
        
        _ = os.system('clear')
    
    def __str__(self):
        
        ret = '\n\n    Remain One\n\n'
        
        pi = 0
        letters = 'abcdefghijklmnopqrstuvxwyz'
        for ri, row in enumerate(Game._grid):
            ret += letters[ri] + ' '
            for slot in row:
                ret += ' '
                if slot is None:
                    ret += ' '
                else:
                    ret += 'o' if self.places[pi].is_full() else '.'
                    pi += 1
            ret += '\n'
        
        ret += '\n   1 2 3 4 5 6 7\n\n'
        
        return ret
    
    def get_state(self):
        ''' Returns state as list of 0s and 1s, 1s representing places with pieces and 0s without. '''
        
        state = []
        for place in self.places:
            state.append(int(place.is_full()))
        
        return state
    
    def first_turn(self, inpt):
        pass
    
    def turn(self, inpt):
        pass
    
    def play(self):
        pass
    
    def str_history(self):
        
        hist = ''
        for i, play in enumerate(self.history): 
            hist += str(i) + '. ' + str(play) + '\n'
        
        return hist


class HumanGame(Game):
    
    _letter_to_number = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6}
    _direction_dictionary = {'up': (0, 1), 'down': (0, -1), 'right': (1, 0), 'left': (-1, 0)}
    
    def in_to_move(self, inpt):
        ''' Takes human input (e.g. 'a4 up') and returns a Move object. '''
        
        ind = (HumanGame._letter_to_number[inpt.split()[0][0]], int(inpt.split()[0][1]) - 1)
        
        direction = HumanGame._direction_dictionary[inpt.split()[1]]
        
        return Move(self, ind, direction)
        
    def first_turn(self, inpt):
        ''' Removes initial piece. If input doesn't make sense, asks for a new one. '''
        
        get_inpt = True
        while get_inpt:
            
            if inpt[0] in 'abcdefg' and inpt[1] in '1234567':
                ind = (HumanGame._letter_to_number[inpt.split()[0][0]], 
                       int(inpt.split()[0][1]) - 1)
                get_inpt = not ((Game._grid.item(ind) is not None) and (ind[0] in range(1, 8) and ind[1] in range(1, 8)))   
            else:
                inpt = input("I don't understand. Choose a piece to remove: ")
                
            if get_inpt:
                inpt = input("That's not a valid piece. Choose a piece to remove: ")
        
        self.history.append(inpt)        
        self.get_place(ind).get_piece().delete()
        
    def turn(self, inpt):
        ''' Makes moves for other turns. If input doesn't make sense, asks for a new one. '''
        
        get_inpt = True
        while get_inpt:

            if inpt[0] in 'abcdefg' and inpt[1] in '1234567' and inpt.split()[1] in ('right', 'left', 'up', 'down'):
                move = self.in_to_move(inpt)
                get_inpt = not move.do()
            else:
                inpt = input("I don't understand. Make a move: ")

            if get_inpt:
                inpt = input('Move not valid. Make another move: ')
        
        self.history.append(inpt)
    
    def play(self):
        ''' Play game. '''
        
        # Remove initial piece
        self.clear_term()
        print(self)
        
        inpt = input('Choose a piece to remove (e.g. "a4"): ')
        self.first_turn(inpt)
        
        # Do other plays from there
        playnum = 1
        while not self.crash():
            
            self.clear_term()
            print(self)
            print('   Play: ' + str(playnum) + '\n')
            
            inpt = input('Write move as: "position direction", e.g. "c4 up".\n' + \
                        'Make your move: ')
            self.turn(inpt)
            
            playnum += 1
        
        # Finish game
        self.clear_term()
        print(self)
        print('   Play: ' + str(playnum) + '\n')
        
        print('Pieces left:', len(self.pieces))
        if len(self.pieces) == 1:
            print('\n   Congratulations!\n')
        else:
            print('\n   Shame... Try again!\n')
        print('\nHistory:\n' + self.str_history())


class ComputerGame(Game):
    ''' Will implement a game in which, at each turn, can take in an input action and return a reward (1 if move is valid and 0 if not, to start with), as well as return the game state as a list. '''
    
    def __init__(self, verbose = False):
        
        super(ComputerGame, self).__init__(verbose)
        
        #self.computer_inds = []
        self.computer_moves = []
        for i in range(Game._grid.shape[0]):
            for j in range(Game._grid.shape[1]):
                if Game._grid[i, j] == 0:
                    #self.computer_inds.append((i, j))
                    for direc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        move = Move(self, (i, j), direc)
                        if move.ever_valid():
                            self.computer_moves.append(move)

        self.direction_dictionary = {(0, 1): 'up', (0, -1): 'down', (1, 0): 'right', (-1, 0): 'left'}
    
        self.history = []
    
    def first_turn(self, inpt):
        ''' Takes in input (int between 0 and 33*4-1) and deletes corresponding piece. Returns reward, which is always 1. '''
        
        self.get_place(self.computer_moves[inpt].ind).get_piece().delete()
        self.history.append(inpt)
        
        return 1
    
    def turn(self, inpt, crash_if_invalid = True):
        ''' Takes in input. Tries to do corresponding move. If valid, does it and returns reward of 1, if not returns reward of 0. '''
        
        valid = self.computer_moves[inpt].do()
        self.history.append(inpt)
        
        if not crash_if_invalid and not valid:
            self.crashed = False
        
        return -10 + 11*int(valid)
    
    def legal_moves(self):
        ''' Returns list with indices corresponding to moves that are legal. '''
        
        legal_moves = []
        for ind, move in enumerate(self.computer_moves):
            if move.is_valid(): legal_moves.append(ind)
        
        return legal_moves
    
    def c_to_h_inpt(self, cinp):
        ''' Given a computer input, returns a human input. '''
        
        move = self.computer_moves[cinp]
        ind = move.ind
        direction = move.direction
        
        return 'abcdefg'[ind[0]] + str(ind[1]+1) + ' ' + self.direction_dictionary[direction]
    
    @classmethod
    def get_inp_out_dims(cls):
        
        return sum(x is not None for x in Game._grid.reshape(-1, )), \
               len(cls().computer_moves)


## Place class
## ~~~~~~~~~~~

class Place(object):
    
    def __init__(self, ind, game, piece = None):
        ''' Ind is place index. '''
        
        self.ind = ind
        self.game = game
        self.piece = piece
    
    def is_full(self):
        return self.piece is not None
    
    def get_piece(self):
        assert self.is_full(), \
            'Place has no piece!'
        return self.piece
    
    def add_piece(self, piece):
        assert not self.is_full(), \
            'Place already has a piece!'
        self.piece = piece
        
    def remove_piece(self):
        assert self.is_full(), \
            'Place already has no piece!'
        self.piece = None


## Piece class
## ~~~~~~~~~~~

class Piece(object):
    
    def __init__(self, game, place):
        
        self.game = game
        self.place = place
        self.ind = self.place.ind
        
        self.place.add_piece(self)
        self.exists = True
        
    def delete(self):
        ''' Deletes itself in self.game.pieces and self.place.piece'''
        
        ind = self.game.get_piece_ind(self)
        
        self.place.piece = None
        del self.game.pieces[ind]
        self.exists = False
        
    def move(self, destination_ind):
        ''' Moves to place in destination index. '''
        
        assert self.exists, \
            "Piece doesn't exist anymore!"
        
        self.place.piece = None
        self.place = self.game.get_place(destination_ind)
        self.place.piece = self
        self.ind = destination_ind


## Move class
## ~~~~~~~~~~

class Move(object):
    
    def __init__(self, game, ind, direction):
        ''' Move has ind and direction attributes (both tuples) so, for example, ind = (1, 2) and direction = (0, -1) moves the piece in the place indexed by (1, 2) down. '''
        
        self.game = game
        self.ind = ind
        self.direction = direction
        
        self.mid_ind = (self.ind[0] - self.direction[1], self.ind[1] + self.direction[0])
        
        self.end_ind = (self.ind[0] - 2*self.direction[1], self.ind[1] + 2*self.direction[0])
        
        assert np.linalg.norm(np.array(direction)) == 1, \
            'Move direction must have norm 1!'
        
    def ever_valid(self):
        ''' Checks if there is a board configuration that could make move valid, i.e. do you land outside the board? '''
        
        for i in range(2):
            if self.end_ind[i] not in range(7):
                if self.game.verbose: print('Destination place is not valid place!')
                return False
            
        if Game._grid.item(self.end_ind) is None:
            if self.game.verbose: print('Destination place is not valid place!')
            return False
        
        return True
        
        
    def is_valid(self):
        ''' Checks if move is valid. '''
        
        # Takes you outside the board
        if not self.ever_valid():
            if self.game.verbose: print('Destination place is not valid place!')
            return False
        
        # Destination place is full
        if self.game.get_place(self.end_ind).is_full():
            if self.game.verbose: print('Destination place is already full!')
            return False
        
        # Starting place is empty
        if not self.game.get_place(self.ind).is_full():
            if self.game.verbose: print('Starting place is not full!')
            return False
        
        # Middle place is empty
        if not self.game.get_place(self.mid_ind).is_full():
            if self.game.verbose: print('Middle place is not full!')
            return False
        
        return True
    
    def do(self):
        ''' Carry out move if valid. Returns True if valid, False otherwise. '''
        
        if not self.is_valid():
            if self.game.verbose: 'Move is not valid!'
            self.game.force_crash()
            return False
        
        start_place = self.game.get_place(self.ind)
        mid_place = self.game.get_place(self.mid_ind)
        
        mid_place.piece.delete()
        start_place.piece.move(self.end_ind)
        
        return True
    


## Play game
## ~~~~~~~~~

if __name__ == "__main__":
    
    HumanGame().play()
    
    
    
        
        
    
    
        
                    
        
    