## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Search game tree for solution
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

''' Simple depth-first game-tree search. Just to check which starting positions work for the games. If it runs for too long, probs no solution as the solution is usually found pretty quickly. '''

# http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=D88E76BC58E895F12AF7F038A603A998?doi=10.1.1.6.4826&rep=rep1&type=pdf

'''

TO CHANGE BOARD SIZE/SHAPE, ONLY NEED TO CHANGE

- BOARD (remember to leave an empty hole already!)
- ROW_RANGE and COL_RANGE

'''


# Set up game
# ~~~~~~~~~~~

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
         [2, 2,  1, 0, 1, 1, 1,  2, 2],
         [2, 2,  1, 1, 1, 1, 1,  2, 2],
         [2, 2,  2, 1, 1, 1, 2,  2, 2],
         [2, 2,  2, 1, 1, 1, 2,  2, 2],

         [2, 2,  2, 2, 2, 2, 2,  2, 2],
         [2, 2,  2, 2, 2, 2, 2,  2, 2]
         ]

ROW_RANGE = []
midc = len(BOARD[0])//2
for r in range(len(BOARD)):
    if BOARD[r][midc] != WALL: ROW_RANGE.append(r)
ROW_RANGE = range(min(ROW_RANGE), max(ROW_RANGE)+1)

COL_RANGE = {}
for r in ROW_RANGE:
    _c = []
    for ci, c in enumerate(BOARD[r]):
        if c != WALL: _c.append(ci)
    COL_RANGE[r] = range(min(_c), max(_c)+1)

VALID   = True  # Validity of moves
INVALID = False

WON  = 1
LOST = 0


## Useful functions
## ~~~~~~~~~~~~~~~~

# Count pegs
def count(board):
    ''' Returns number of pegs in board. '''
    
    n_pegs = 0
    for r in ROW_RANGE:
        for c in COL_RANGE[r]:
            if board[r][c] is FULL: n_pegs += 1
    
    return n_pegs

# Test validity of move
def move(piece, dir, board):
    ''' Assumes starting place is full. Returns VALID if middle place is full and end place empty. '''
    
    mid_place   = board[piece[0] + dir[0]][piece[1] + dir[1]]
    if mid_place is not FULL: return INVALID
    
    end_place   = board[piece[0] + 2*dir[0]][piece[1] + 2*dir[1]]
    if end_place is not FREE: return INVALID
    
    return VALID

# Make a move
def do_move(piece, dir, board):
    ''' Assumes move is valid. Empties initial place and middle place and fills up end place. '''
    
    board[piece[0]][piece[1]] = FREE
    board[piece[0] + dir[0]][piece[1] + dir[1]] = FREE
    board[piece[0] + 2*dir[0]][piece[1] + 2*dir[1]] = FULL
    
    return board

# Undo a move
def undo_move(piece, dir, board):
    ''' Assumes move was just done. Fills up initial place and middle place and empties end place. '''
    
    board[piece[0]][piece[1]] = FULL
    board[piece[0] + dir[0]][piece[1] + dir[1]] = FULL
    board[piece[0] + 2*dir[0]][piece[1] + 2*dir[1]] = FREE
    
    return board

# Test for victory
def test_possibilities(board, memory, num_pegs, spaces = ''):
    ''' Recursively checks (first) whether game is won and (second) for any valid move given a game position. 
        Returns WON if game is won, LOST if no valid moves. '''
    
    # Check for victory
    if num_pegs == 1:
        return memory
    
    # Check for valid moves
    no_valid_moves = True
    for r in ROW_RANGE:
        for c in COL_RANGE[r]:
            
            print('Checking (%i, %i)...' % (r, c))
            
            if board[r][c] is FULL:
                for direc in DIRS:
                    
                    print('  in direction', direc)
                    
                    if move((r, c), direc, board) is VALID:
                        
                        no_valid_moves = False
                        
                        # Do move, add to memory and recursively check for victory
                        board = do_move((r, c), direc, board)
                        memory.append(((r,c), direc))
                        
                        result = test_possibilities(board, memory.copy(), num_pegs - 1)
                        if result is not None: 
                            return result
                        
                        # Dead end - remove from memory and undo move
                        memory = memory[:-1]
                        board = undo_move((r, c), direc, board)
    
    return None

# Print board for the end
def print_board(board):
    
    r_nums = '  '
    for i, _ in enumerate(ROW_RANGE):
        r_nums += str(i)
    r_nums += '  '

    ret = '\n\n       Remain One\n\n'
        
    pi = 0
    for ri, row in enumerate(board):
        ret += r_nums[ri] + ' '
        for slot in row:
            ret += ' '
            if slot is WALL:
                ret += ' '
            else:
                ret += 'o' if slot is FULL else '.'
                pi += 1
        ret += '\n'
    
    minc = 99
    maxc = 0
    for r in ROW_RANGE:
        minc = min(COL_RANGE[r]) if min(COL_RANGE[r]) < minc else minc
        maxc = max(COL_RANGE[r]) if max(COL_RANGE[r]) > maxc else maxc
            
    ret += '\n       '
    ret += ' '.join([str(i) for i in range(maxc-minc+1)])
    ret += '     \n\n'
        
    print(ret)


## Do it
## ~~~~~

if __name__ == "__main__":
    
    import time, matplotlib.pyplot as plt, os
    
    # Start relevant objects
    num_pegs = count(BOARD)
    
    print('Start with', num_pegs)
    
    # Calculate (and time)
    start_time = time.time()
    print('Starting search...')
    memory = test_possibilities([row.copy() for row in BOARD], [], num_pegs)
    print('Done calculating in %.1f minutes.' \
            % ((time.time() - start_time)/60))
    
    # Print results
    board = BOARD
    for m, move in enumerate(memory):
        
        os.system('clear')
        print(move[0], {(-1, 0): 'UP', (1, 0): 'DOWN', (0, -1): 'LEFT', (0, 1): 'RIGHT'}[move[1]])
        print('\nMove', m, 'with', count(board), 'pieces left.')
        print_board(board)
        
        board = do_move(move[0], move[1], board)
        time.sleep(0.5)
    
    os.system('clear')
    print(move[0], {(-1, 0): 'UP', (1, 0): 'DOWN', (0, -1): 'LEFT', (0, 1): 'RIGHT'}[move[1]])
    print('\nMove', len(memory)+1, 'with', count(board), 'pieces left.')
    print_board(board)
    
    input('Press Enter to see move history.')
    for move in memory:
        print(move[0], {(-1, 0): 'UP', (1, 0): 'DOWN', (0, -1): 'LEFT', (0, 1): 'RIGHT'}[move[1]])
        
    
                        
                    
    
    
    








