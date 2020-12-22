## ~~~~~~~~~~~~~~~~~~
#  nV-learning player
## ~~~~~~~~~~~~~~~~~~

'''

n-step V-learning player. Instead of training on reward of action taken in current state plus discounted value of next state, trains on the discounted sum of the actual future actions in the same game.


'''

import numpy as np, math
from .vplayer import *

class nVPlayer(VPlayer):
    ''' Only need to rewrite init, store and train methods. 
    
        Relies on sequential stores being chronological states of the same game and crash == True separating different games (demarcating the end of a game). 
        
        Initialises memory of infinite size but stores actual max memory len. Stores without thought to memory len until a crash == True example comes in, when it does the addition of rewards (in order to have this be n-step learning) and erases the overload from the beginning of the memory. train just draws a batch and trains - no fancy stuff. 
        '''
    
    def __init__(self, disc_rate, max_memory_len, identifier):
        
        super().__init__(self, disc_rate, math.inf, identifier)
        self.max_memory_len = max_memory_len
        
        self.game_len = 0
        
    
    def store(self, state, action, reward, new_state, crash, legal_moves):
        ''' No need to store action, future state or legal moves. '''
        
        self.memory.append([state, reward, crash], 
                           multiple = False)
        self.game_len += 1
        
        if crash is True:
            
            R = 0
            for i in range(1, self.game_len+1):
                R = self.memory[-i][1] + self.disc_rate * R
                self.memory[-i][1] = R
            
            self.game_len = 0
            
            overload = len(self.memory) - self.max_memory_len
            if overload > 0:
                self.memory.memory = self.memory.memory[overload:]
        
    def train(self, train_set_size, batch_size, l_rate, 
              reg_rate = 0.01, mom_rate = 0.85,
              use_last = False, verbose = False):
        ''' Trains on batch drawn randomly from memory. '''
        
        # Gen batch for training
        if train_set_size == -1:
            train_set = np.array(self.memory.memory)
        elif use_last:
            train_set = np.array(self.memory.memory[-min(len(self.memory), train_set_size):])
        else:
            train_set = np.random.permutation(self.memory)[:train_set_size]
        
        if train_set_size == -1:
            train_set_size = len(train_set)
        else:
            train_set_size = min(len(train_set), train_set_size)
        
        # Training examples and targets
        examples = np.array(list(train_set[:, 0]))
        targets = np.array(list(train_set[:, 1]))
        
        # Train
        if verbose: old_fp = self.net_predict(examples)
        
        self.net_train(examples, targets, 
                       num_iterations = 1, 
                       batch_size = batch_size,
                       l_rate = l_rate,
                       mom_rate = mom_rate,
                       reg_rate = reg_rate)
        
        # Print stuff
        if verbose:
            
            new_fp = self.net_predict(examples)
            
            print('  RMSE:', np.mean((old_fp - targets)**2)**0.5, 
                  '->', np.mean((new_fp - targets)**2)**0.5,
                  '\n  Cost:', self.network.cost(old_fp, targets),
                        '->', self.network.cost(new_fp, targets))
        