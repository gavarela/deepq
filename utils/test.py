


import numpy as np, random, time, json, matplotlib.pyplot as plt
from WillyNet import WillyNet
#from sklearn import MLPRegressor


## Params
## ~~~~~~

DISC_RATE = 0.95
MAX_MEM_LEN = 1000

BATCH_SIZE = 70
L_RATE = 0.02
REG_RATE = 0.0001
MOM_RATE = 0.8

NUM_EPOCHS = 200
VERBOSE = 1

EPSILON = lambda i: 1 - i/NUM_EPOCHS

MAX_TURNS = 100


## Game
## ~~~~

class Game(object):
    
    _moves = [1, -1]
    
    def __init__(self):
        
        self.food_pos = random.choice((0, 10))
        self.player_pos = 5
        
        self.crashed = False
        self.won = False
        
    def turn(self, move_ind):
        
        self.player_pos += Game._moves[move_ind]
        
        # If out of board to left, end
        if self.player_pos not in range(11):
            self.crashed = True
            return -1
            
        # If got food, done
        elif self.player_pos == self.food_pos:
            self.won = True
            return 1
        
        return 0
        
    def get_state(self):
        
        state = np.zeros(2*11)
        if self.player_pos in range(11):
            state[self.player_pos] = 1
        state[11 + self.food_pos] = 1
        
        return state
    
    def restart(self):
        
        self.__init__()


## (Q-)Player
## ~~~~~~~~~~

class QPlayer(object):
    
    def __init__(self, max_mem_len, disc_rate):
        
        self.max_mem_len = max_mem_len
        self.disc_rate = disc_rate
        
        self.memory = []
        
        self.network = WillyNet(shape = [22, 100, 100, 2], 
                                problem = 'regression')
#        self.network = MLPRegressor((100, 100), 'sgd', 
#                                    alpha = REG_RATE,
#                                    learning_rate_init = LEARN_RATE,
#                                    momentum = MOM_RATE,
#                                    batch_size = BATCH_SIZE)
    
    def store(self, state, action, reward, future_state, crashed):
        
        item = [state, action, reward, future_state, crashed]
        
        if len(self.memory) == self.max_mem_len:
            self.memory = self.memory[1:]
        
        self.memory.append(item)
        
    def get_action(self, state, at_random):
        
        if at_random:
            ret = random.choice(range(self.network.shape[-1]))
#            print('Random choice in get_action:', ret)
            return ret
        else:
            ret = self.network.predict(np.array([state]), pred_type = 'argmax')[0][0]
#            print('Non-random get_action:', self.network.predict(np.array([state])), '->', ret)
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
        for example in batch:
            
            # Build target:
            # 0: state, 1: action, 2: reward, 
            # 3: new state, 4: crashed
            target = self.network.predict(np.array([example[0]]))[0]
            target[example[1]] = example[2]
            
            if not example[4]:
                target[example[1]] += self.disc_rate * max(self.network.predict(np.array([example[3]]))[0])
            
            # Append to lists
            examples.append(example[0])
            targets.append(target)
            
        # Train
        self.network.train(np.array(examples), np.array(targets),
                           l_rate, 1, reg_rate, 1, mom_rate)



## Play game (computer)
## ~~~~~~~~~

def play_hgame(game):
    
    while not (game.crashed or game.won):
                
        line = '\nBoard:\n  |  '
        for ind in ['x' if i == game.food_pos else \
                    'A' if i == game.player_pos else \
                    '-' for i in range(11)]:
            line += ind
        line += '  |\n'
        print(line)
        
        print('\nState:', game.get_state())
        
        action = None
        while action not in ('0', '1'):
            action = input('\nEnter move: ')

        print('\nReward:', game.turn(int(action)))


## Play game (computer)
## ~~~~~~~~~

def play_cgame(game, player, epsilon):
    
    turns = 0
    new_state = game.get_state()
    while (not (game.crashed or game.won)) and (turns < MAX_TURNS):

        # Play turn
        state = new_state
        action = player.get_action(state, 
                                   random.random() < epsilon)
        reward = game.turn(action)
        new_state = game.get_state()
        
#        print('Did turn:', state, action, reward, '\n         ', new_state, game.crashed, '\n')

        turns += 1
        
        # Store in memory
        player.store(state, action, reward, new_state, game.crashed)
    
#    input('\nEnd of game. Press Enter to continue to next one.')
    return turns


## Do RL
## ~~~~~

if __name__ == "__main__":
    
    print('\033[33m')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('\n       Test - Simple Game       \n')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('\033[0m')
    
    game = Game()
    
    
    # Play game - human
    # ~~~~~~~~~~~~~~~~~
    
    if input('Enter anything to play the game: ') != '':
        
        print('\nWelcome to the game. Each turn, press 0 to move to the right and 1 to move to the left. Objective is to fall out of the right of the map.')
        
        while True:
            
            game.restart()
            
            play_hgame(game)
                
            if game.won:
                print('\nCongrats! You won :)')
            else:
                print('\nOh no! You lost :(')
            
            if input('Enter anything to play again: ') == '':
                break
    
    
    # Reinforcement learning
    # ~~~~~~~~~~~~~~~~~~~~~~
    
    player = QPlayer(MAX_MEM_LEN, DISC_RATE)
    
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
          '\n - epochs              =', NUM_EPOCHS, '\n')
    
    turns = play_cgame(game, player, -1)
    print('\nPlayed deterministic game before training. Lasted %i turns and %s' %(turns, 'won!' if game.won else 'lost...'))
    
    turnlist = []
    winlist = []
    
    det_turnlist = []
    det_winlist = []
    
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        
        # Play a game
        game.restart()
        turns = play_cgame(game, player, EPSILON(epoch))
        
        turnlist.append(turns)
        winlist.append(game.won)
        
        print('\nEpoch %i: played %i turns and %s' %(epoch+1, turns, 'won!' if game.won else 'lost...'))
        
        # Train
        player.train(BATCH_SIZE, L_RATE, REG_RATE, MOM_RATE)
        
        # If right epoch, play deterministic game
        if (epoch+1) % VERBOSE == 0:
            
            game.restart()
            turns = play_cgame(game, player, -1)
            print('\n  Played deterministic game %i minutes into training.\n    Lasted %i turns and %s' %((time.time() - start_time) / 60, turns, 'won!' if game.won else 'lost...'))
            det_turnlist.append(turns)
            det_winlist.append(game.won)
            
    # All done, save network and turn list and stuff
    player.network.save('network.json')
    
    data = {'turnlist': turnlist,
            'winlist': winlist,
            'det_turnlist': det_turnlist,
            'det_winlist': det_winlist}
    file = open('results.json', 'w')
    json.dump(data, file)
    file.close()
    
    # Plot
    fig, ax = plt.subplots()
    
    for i in range(len(det_turnlist)):
        ax.plot(i * VERBOSE, det_turnlist[i], 
                'g.' if det_winlist[i] else 'r.',
                markersize = 1)
    
    plt.show(block = False)
    input('PE.')
    plt.close()
            
        
        
        
    
    
    
    
    
    
    
    
    
    
        
    
    

        
    