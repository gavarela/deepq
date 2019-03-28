## ~~~~~~~~~~~~~~~~~~
#  Testing Q-Learning
## ~~~~~~~~~~~~~~~~~~


'''

Simple game. Build grid and randomly place player, food and (if set to) pits. Want to get to food without falling in pits or off the grid. Easy to solve when small. Gets harder fast...

Keeping all other params the same (bar VERBOSE ones):
B = 3, P = 0: 1.2k epochs
B = 3, P = 1: 2.5k epochs
B = 3, P = 2: 4k epochs
B = 4, P = 0: 20k epochs
B = 4, P = 1: 
B = 5, P = 0: 

'''

import numpy as np, random, json
from collections import deque

from WillyNet import WillyNet


## Game
## ~~~~

class Game(object):
    
    _moves = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    
    def __init__(self, n_pits = 3, board_size = 5):
        
        self.n_pits = n_pits
        self.board_size = board_size
        
        self.food_pos = [random.choice(range(board_size)),
                         random.choice(range(board_size))]
        
        self.player_pos = [random.choice(range(board_size)),
                           random.choice(range(board_size))]
        while self.player_pos == self.food_pos:
            self.player_pos = [random.choice(range(board_size)),
                               random.choice(range(board_size))]
        
        self.other_pos = (self.food_pos, self.player_pos)
        self.pits_pos = []
        for pit in range(n_pits):
            self.pits_pos.append([random.choice(range(board_size)),
                                  random.choice(range(board_size))])
            while self.pits_pos[-1] in self.other_pos or \
                    self.pits_pos[-1] in self.pits_pos[:-1]:
                self.pits_pos[-1] = [random.choice(range(board_size)),
                                     random.choice(range(board_size))]
        
        self.crashed = False
        self.won = False
        
    def turn(self, move_ind):
        
        self.player_pos[0] += Game._moves[move_ind][0]
        self.player_pos[1] += Game._moves[move_ind][1]
        
        # If out of board, end
        if self.player_pos[0] not in range(self.board_size) or \
            self.player_pos[1] not in range(self.board_size):
            self.crashed = True
            return -1
        
        # If fell in pit, end
        elif self.player_pos in self.pits_pos:
            self.crashed = True
            return -1
            
        # If got food, done
        elif self.player_pos == self.food_pos:
            self.won = True
            return 1
        
        return 0
        
    def get_state(self):
        
        state = np.zeros((2 + (self.n_pits > 0))*self.board_size**2)
        
        # Add player
        if not self.crashed:
            state[self.board_size*self.player_pos[0] + \
                  self.player_pos[1]] = 1
        
        # Add food
        state[self.board_size**2 + \
              self.board_size*self.food_pos[0] + \
              self.food_pos[1]] = 1
        
        # Add pits
        for pit_pos in self.pits_pos:
            state[2*self.board_size**2 + \
                  self.board_size*pit_pos[0] + \
                  pit_pos[1]] = 1
        
#        state = self.player_pos + self.food_pos
#        for pit_pos in self.pits_pos:
#            state += self.pit_pos
        
        return state
    
    def restart(self):
        
        self.__init__(self.n_pits, self.board_size)


## (Q-)Player
## ~~~~~~~~~~

class QPlayer(object):
    
    def __init__(self, max_mem_len, disc_rate, shape):
        
        self.max_mem_len = max_mem_len
        self.disc_rate = disc_rate
        
        self.memory = deque()
        
        self.network = WillyNet(shape = shape, 
                                problem = 'regression')
    
    def store(self, *args):
        
        item = list(args)
        
        if len(self.memory) == self.max_mem_len:
            self.memory.popleft()
        
        self.memory.append(item)
        
    def get_action(self, state, at_random):
        
        if at_random:
            ret = random.choice(range(self.network.shape[-1]))
            #print('Random choice in get_action:', ret)
            return ret
        else:
            ret = self.network.predict(np.array([state]), pred_type = 'argmax')[0][0]
            #print('Non-random get_action:', self.network.predict(np.array([state])), '->', ret)
            return ret
    
    def train(self, batch_size, l_rate, reg_rate, mom_rate):
        
        # Get batch
        if batch_size < len(self.memory):
            batch = np.array(random.sample(self.memory, batch_size))
        else:
            batch = self.memory
        
        # Build examples and targets
        examples = []
        targets = []
        for state, action, reward, new_state, crashed in batch:
            
            # Build target
            target = self.network.predict(np.array([state]))[0]
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

def play_hgame(game):
    
    game.restart()
    print('Starting game with:', game.other_pos, game.pits_pos)
    
    while not (game.crashed or game.won):
                
        print('\nBoard:\n')
        for row in range(BOARD_SIZE):
            line = '  |  '
            for char in ['x' if [row, i] == game.food_pos else \
                         'O' if [row, i] in game.pits_pos else \
                         'A' if [row, i] == game.player_pos else \
                         '.' for i in range(BOARD_SIZE)]:
                line += char
            line += '  |'
            print(line)
        
        print('State is raveled this:', '\n', np.array(game.get_state()).reshape((2 + (N_PITS>0), BOARD_SIZE, BOARD_SIZE)))
        
        action = None
        while action not in ('0', '1', '2', '3'):
            action = input('\nEnter move: ')

        print('\nReward:', game.turn(int(action)))


## Play game (computer)
## ~~~~~~~~~

def play_cgame(game, player, epsilon):
    
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

    NUM_EPOCHS = 700
    EPSILON = lambda i: 1 - i/NUM_EPOCHS

    VERBOSE = 10
    DET_VERBOSE = 5
    NUM_DET_GAMES = 15
    MAX_DET_TRIES = 50

    MAX_TURNS = 100
    
    
    game = Game(N_PITS, BOARD_SIZE)
    
    
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
    
    player = QPlayer(MAX_MEM_LEN, DISC_RATE, SHAPE)
    
    turnlist = []
    winlist = []
    
    det_turnlist = []
    det_winlist = []
    
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        
        # Play a game
        turns, won = play_cgame(game, player, EPSILON(epoch))
        
        turnlist.append(turns)
        winlist.append(game.won)
        
        if (epoch+1) % VERBOSE == 0:
            print('\nEpoch %i: played %i turn(s) and %s' %(epoch+1, turns, 'won!' if won else 'lost...'))
        
        # Train
        player.train(BATCH_SIZE, L_RATE, REG_RATE, MOM_RATE)
        
        # If right epoch, play deterministic game
        if (epoch+1) % DET_VERBOSE == 0:
            
            print('\n  Det games:')
            
            # Play three that don't get stuck
            n_wins = 0
            non_stuck = 0
            tries = 0
            while non_stuck < NUM_DET_GAMES and tries < MAX_DET_TRIES:
                turns, won = play_cgame(game, player, -1)
                
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
    
    print('\nDone running RL! Saving...')
    
    # All done, save network and turn list and stuff
    player.network.save('test_saves/network_%i_%i.json'  %(BOARD_SIZE, N_PITS))
    
    data = {'turnlist': turnlist,
            'winlist': winlist,
            'det_turnlist': det_turnlist,
            'det_winlist': det_winlist,
            'params': {'N_PITS': N_PITS,
                       'BOARD_SIZE': BOARD_SIZE,
                       'DISC_RATE': DISC_RATE,
                       'MAX_MEM_LEN': MAX_MEM_LEN,
                       'BATCH_SIZE': BATCH_SIZE,
                       'L_RATE': L_RATE,
                       'REG_RATE': REG_RATE,
                       'MOM_RATE': MOM_RATE,
                       'NUM_EPOCHS': NUM_EPOCHS,
                       'DET_VERBOSE': DET_VERBOSE,
                       'NUM_DET_GAMES': NUM_DET_GAMES,
                       'MAX_DET_TRIES': MAX_DET_TRIES,
                       'MAX_TURNS': MAX_TURNS}
            }
    
    file = open('test_saves/results_%i_%i.json'  %(BOARD_SIZE, N_PITS), 'w')
    json.dump(data, file)
    file.close()
    
    # Plot
    print('Saved. Plotting...')
    fig, ax = plt.subplots()
    
    for i in range(len(det_winlist)):
        ax.plot(i * DET_VERBOSE, det_winlist[i], 
                'g.' if det_winlist[i] > (NUM_DET_GAMES//2) else 'r.',
                markersize = 1)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Games won (of %i)' %NUM_DET_GAMES)
    ax.set_title('(%i, %i) w/ %i pit(s)' %(BOARD_SIZE, BOARD_SIZE,
                                           N_PITS))
    
    fig.suptitle('RL Progress')
    
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
    plt.savefig('test_saves/plots/progress_%i_%i.pdf' %(BOARD_SIZE, N_PITS))
    input('PE.')
    plt.close()
            
        
        
        
    
    
    
    
    
    
    
    
    
    
        
    
    

        
    